from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import Settings


SourceType = Literal["local_docs", "teknofest_site", "tavily"]


@dataclass
class RetrievedChunk:
    content: str
    metadata: Dict[str, Any]
    score: float | None
    source_type: SourceType


def _build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
    )


def _build_chroma_collection(
    settings: Settings,
    path,
    collection_name: str,
) -> Chroma:
    embeddings = _build_embeddings(settings)
    client = PersistentClient(path=str(path))
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
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
    return TavilySearchResults(api_key=settings.tavily_api_key, max_results=5)


def retrieve_from_vectorstore(
    vs: Chroma,
    query: str,
    source_type: SourceType,
    k: int = 5,
) -> List[RetrievedChunk]:
    docs_and_scores = vs.similarity_search_with_score(query, k=k)
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
    return chunks


def estimate_chunk_confidence(chunks: List[RetrievedChunk]) -> float:
    """
    Chroma score'u distance tabanlıdır (küçük skor daha iyi).
    Basit normalize edilmiş confidence skoru üretir: [0.0, 1.0]
    """
    if not chunks:
        return 0.0
    values = []
    for chunk in chunks:
        if chunk.score is None:
            continue
        values.append(max(0.0, min(1.0, 1.0 - float(chunk.score))))
    if not values:
        return 0.0
    return sum(values) / len(values)


def retrieve_from_tavily(
    tool: TavilySearchResults,
    query: str,
) -> List[RetrievedChunk]:
    raw_results = tool.invoke({"query": query})
    chunks: List[RetrievedChunk] = []
    for item in raw_results:
        content = item.get("content") or item.get("snippet") or ""
        chunks.append(
            RetrievedChunk(
                content=content,
                metadata={
                    "url": item.get("url"),
                    "title": item.get("title"),
                },
                score=None,
                source_type="tavily",
            )
        )
    return chunks

