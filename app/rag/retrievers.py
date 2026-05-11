"""
retrievers.py
=============
Retrieval layer — vector store construction and top-K similarity search.

All Chroma instances use the centralized EmbeddingService.
No embedding logic lives here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from chromadb import PersistentClient

try:
    from langchain_chroma import Chroma
except ImportError:  # fallback for older installs
    from langchain_community.vectorstores import Chroma  # type: ignore[assignment]

try:
    from langchain_tavily import TavilySearch as TavilySearchResults  # noqa: F401
except ImportError:  # fallback for older installs
    from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore[assignment]

from app.config import Settings
from app.rag.embedding_service import get_embedding_service

from langsmith import traceable

logger = logging.getLogger(__name__)

SourceType = Literal["local_docs", "teknofest_site", "tavily"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with content, metadata, and scoring info."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    source_type: SourceType = "local_docs"

    # Convenience properties
    @property
    def source(self) -> str:
        return (
            self.metadata.get("source")
            or self.metadata.get("crawl_source")
            or self.metadata.get("url")
            or "unknown"
        )

    @property
    def page(self) -> Optional[int]:
        return self.metadata.get("page_number")

    @property
    def section(self) -> Optional[str]:
        return self.metadata.get("section_title")

    @property
    def source_priority(self) -> int:
        return int(self.metadata.get("source_priority", 1))


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------


def _build_chroma_collection(
    settings: Settings,
    path,
    collection_name: str,
) -> Chroma:
    """Build a Chroma vectorstore using the centralized EmbeddingService."""
    lc_embeddings = get_embedding_service(settings).get_langchain_embeddings()
    client = PersistentClient(path=str(path))
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=lc_embeddings,
    )


def build_local_docs_retriever(settings: Settings) -> Chroma:
    return _build_chroma_collection(
        settings=settings,
        path=settings.chroma_local_docs_path,
        collection_name=settings.chroma_local_collection,
    )


def build_teknofest_site_retriever(settings: Settings) -> Chroma:
    return _build_chroma_collection(
        settings=settings,
        path=settings.chroma_teknofest_site_path,
        collection_name=settings.chroma_site_collection,
    )


def build_tavily_tool(settings: Settings) -> TavilySearchResults:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    
    kwargs = {"api_key": settings.tavily_api_key, "max_results": 5}
    
    if getattr(settings, "tavily_use_domain_filter", False) and getattr(settings, "tavily_trusted_domains", None):
        kwargs["include_domains"] = settings.tavily_trusted_domains
        kwargs["max_results"] = 8  # Use more results when filtering by domain
        
    try:
        # New langchain-tavily package
        return TavilySearchResults(**kwargs)  # type: ignore[call-arg]
    except TypeError:
        # Older langchain-community signature
        return TavilySearchResults(**kwargs)


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


@traceable(
    run_type="retriever",
    name="vectordb_retrieval",
    metadata={
        "pipeline_stage": "retrieval",
        "source": "chromadb"
    }
)
def retrieve_from_vectorstore(
    vs: Chroma,
    query: str,
    source_type: SourceType,
    k: int = 10,
) -> List[RetrievedChunk]:
    """
    Retrieve top-K chunks from a Chroma vectorstore.

    Returns structured RetrievedChunk objects with full metadata
    and similarity distance scores.
    """
    try:
        docs_and_scores = vs.similarity_search_with_score(query, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vectorstore retrieval failed (%s): %s", source_type, exc)
        return []

    chunks: List[RetrievedChunk] = []
    for doc, score in docs_and_scores:
        chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                metadata=doc.metadata or {},
                score=float(score),
                source_type=source_type,
            )
        )

    logger.debug("Retrieved %d chunks from %s (k=%d)", len(chunks), source_type, k)
    return chunks


@traceable(
    run_type="tool",
    name="tavily_search",
    metadata={"pipeline_stage": "live_fetch"}
)
def retrieve_from_tavily(
    tool: TavilySearchResults,
    query: str,
) -> List[RetrievedChunk]:
    """Retrieve from Tavily web search."""
    TRUSTED_DOMAINS = ["teknofest.org", "t3vakfi.org", "trthaber.com", "aa.com.tr"]
    
    try:
        raw_results = tool.invoke({"query": query, "include_domains": TRUSTED_DOMAINS})
    except Exception:
        domain_filter = " OR ".join([f"site:{d}" for d in TRUSTED_DOMAINS])
        try:
            raw_results = tool.invoke(f"{query} ({domain_filter})")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily retrieval failed: %s", exc)
            return []

    chunks: List[RetrievedChunk] = []
    items = raw_results.get("results", []) if isinstance(raw_results, dict) else raw_results
    
    def domain_rank(url: str) -> int:
        for i, domain in enumerate(TRUSTED_DOMAINS):
            if domain in url:
                return i
        return 99

    items.sort(key=lambda r: domain_rank(r.get("url", "")))
    trusted_items = [r for r in items if domain_rank(r.get("url", "")) < 99]
    items = trusted_items if trusted_items else items
    
    for item in items:
        content = item.get("content") or item.get("snippet") or ""
        if not content.strip():
            continue
        chunks.append(
            RetrievedChunk(
                content=content,
                metadata={
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "source": item.get("url"),
                    "document_type": "web",
                    "source_priority": 1,
                },
                score=None,
                source_type="tavily",
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# A.3 — Startup collection health check
# ---------------------------------------------------------------------------


def verify_collections(settings: Settings) -> dict:
    """
    Startup sırasında her iki Chroma collection'ının sağlığını doğrular.
    Chunk sayısı, cosine metric durumu ve embedding boyutunu raporlar.
    """
    results = {}
    collection_map = [
        ("local_docs", settings.chroma_local_docs_path, settings.chroma_local_collection),
        ("teknofest_site", settings.chroma_teknofest_site_path, settings.chroma_site_collection),
    ]
    for name, path, col_name in collection_map:
        try:
            client = PersistentClient(path=str(path))
            cols = client.list_collections()
            found = [c for c in cols if c.name == col_name]
            if not found:
                results[name] = {
                    "status": "missing",
                    "chunks": 0,
                    "collections": [c.name for c in cols],
                    "cosine": False,
                }
                continue
            col = found[0]
            count = col.count()
            meta = col.metadata or {}
            cosine_ok = meta.get("hnsw:space") == "cosine"
            results[name] = {
                "status": "ok",
                "chunks": count,
                "cosine": cosine_ok,
                "embedding_model": meta.get("embedding_model", "unknown"),
                "collections": [c.name for c in cols],
            }
        except Exception as e:  # noqa: BLE001
            results[name] = {"status": "error", "error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------


def estimate_chunk_confidence(chunks: List[RetrievedChunk], settings: Optional[Settings] = None) -> float:
    """
    Convert distance-based Chroma scores to a confidence value in [0, 1].

    Uses Top-N average and a hard floor for high-quality single chunks
    to ensure Local RAG is prioritized when a highly relevant chunk exists.
    """
    if not chunks:
        return 0.0

    # Default parameters if settings not provided
    hard_floor_score = settings.rag_hard_floor_score if settings else 0.45
    hard_floor_confidence = settings.rag_hard_floor_confidence if settings else 0.56
    top_n = settings.rag_top_n_for_confidence if settings else 3

    values = []
    best_similarity = 0.0

    for chunk in chunks:
        if chunk.score is None:
            continue
        # Score is distance: 0 = identical, 2 = opposite direction
        # Map distance 0→2 to confidence 1→0
        similarity = max(0.0, min(1.0, 1.0 - (float(chunk.score) / 2.0)))
        values.append(similarity)
        
        if similarity > best_similarity:
            best_similarity = similarity

    if not values:
        return 0.0

    # --- Hard Floor Check ---
    if best_similarity >= hard_floor_score:
        return max(hard_floor_confidence, best_similarity)

    # Sort values descending (highest similarity first)
    sorted_values = sorted(values, reverse=True)

    # --- Q10 Expert Advice: Gap Rule ---
    # Accept top candidate if the gap between top1 and top2 is significant
    if len(sorted_values) >= 2:
        gap = sorted_values[0] - sorted_values[1]
        if gap >= 0.15:
            return max(hard_floor_confidence, best_similarity)

    # --- Top-N Average ---
    top_n_values = sorted_values[:top_n]

    return sum(top_n_values) / len(top_n_values)
