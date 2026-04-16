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

try:
    from langsmith import traceable as _ls_traceable
    _retrieval_step = _ls_traceable(name="retrieval_step", run_type="retriever")
except ImportError:  # pragma: no cover
    def _retrieval_step(fn):  # type: ignore[misc]
        return fn

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
        collection_name="local_docs",
    )


def build_teknofest_site_retriever(settings: Settings) -> Chroma:
    return _build_chroma_collection(
        settings=settings,
        path=settings.chroma_teknofest_site_path,
        collection_name="teknofest_site",
    )


def build_tavily_tool(settings: Settings) -> TavilySearchResults:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    try:
        # New langchain-tavily package
        return TavilySearchResults(api_key=settings.tavily_api_key, max_results=5)  # type: ignore[call-arg]
    except TypeError:
        # Older langchain-community signature
        return TavilySearchResults(api_key=settings.tavily_api_key, max_results=5)


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


@_retrieval_step
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


@_retrieval_step
def retrieve_from_tavily(
    tool: TavilySearchResults,
    query: str,
) -> List[RetrievedChunk]:
    """Retrieve from Tavily web search."""
    try:
        raw_results = tool.invoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tavily retrieval failed: %s", exc)
        return []

    chunks: List[RetrievedChunk] = []
    items = raw_results.get("results", []) if isinstance(raw_results, dict) else raw_results
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
# Confidence estimation
# ---------------------------------------------------------------------------


def estimate_chunk_confidence(chunks: List[RetrievedChunk]) -> float:
    """
    Convert distance-based Chroma scores to a confidence value in [0, 1].

    Chroma returns L2 or cosine distances (lower = more similar).
    We invert and clamp to produce a confidence score.
    """
    if not chunks:
        return 0.0
    values = []
    for chunk in chunks:
        if chunk.score is None:
            continue
        # Score is distance: 0 = identical, 2 = opposite direction
        # Map distance 0→1 to confidence 1→0
        confidence = max(0.0, min(1.0, 1.0 - float(chunk.score)))
        values.append(confidence)
    if not values:
        return 0.0
    return sum(values) / len(values)
