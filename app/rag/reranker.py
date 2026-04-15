"""
reranker.py
===========
Optional LLM-based reranking of retrieved chunks.

Controlled by ``settings.reranker_enabled`` (ENABLE_RERANKING env var).
When disabled, the original retrieval order is preserved.

Design
------
A single batch prompt is issued to the LLM asking it to score each chunk
on a 0-10 relevance scale.  This avoids N separate LLM calls and keeps
latency reasonable.

Fallback
--------
If the LLM call fails or scores cannot be parsed, the original order is
returned unchanged so the pipeline never breaks due to reranking errors.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from app.rag.retrievers import RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_RERANK_PROMPT_TEMPLATE = """You are a relevance scoring engine for a RAG system about TEKNOFEST.

Given the user's query and a list of text chunks, score each chunk on its
relevance to the query on a scale from 0 (completely irrelevant) to 10 (perfectly relevant).

Return ONLY a comma-separated list of scores in the same order as the chunks.
Example for 4 chunks: 8,3,9,1

Query:
{query}

Chunks:
{chunks}

Scores (comma-separated, one per chunk, same order):"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def rerank_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    llm: BaseChatModel,
    final_k: int,
) -> List[RetrievedChunk]:
    """
    Rerank *chunks* by relevance to *query* using a single LLM batch call.

    Parameters
    ----------
    query : str
        The user's original question.
    chunks : List[RetrievedChunk]
        Top-K candidates from the vector store.
    llm : BaseChatModel
        Any LangChain chat model (typically the same one used for generation).
    final_k : int
        Number of top chunks to return after reranking.

    Returns
    -------
    List[RetrievedChunk]
        Up to *final_k* chunks, reordered by LLM-assigned relevance score.
        Each chunk has a ``rerank_score`` key injected into its metadata.
    """
    if not chunks:
        return []

    # If only one chunk or final_k >= len, skip expensive LLM call
    if len(chunks) <= 1 or final_k >= len(chunks):
        for i, ch in enumerate(chunks):
            ch.metadata["rerank_score"] = 10 - i  # preserve order marker
        return chunks[:final_k]

    scores = await _score_chunks(query, chunks, llm)
    ranked = _apply_scores(chunks, scores)
    return ranked[:final_k]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _score_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    llm: BaseChatModel,
) -> List[float]:
    """Issue one LLM call and return a score per chunk (fallback: equal scores)."""
    chunk_texts = "\n\n".join(
        f"[{i + 1}] {ch.content[:400]}" for i, ch in enumerate(chunks)
    )
    prompt = _RERANK_PROMPT_TEMPLATE.format(query=query, chunks=chunk_texts)

    try:
        res = await llm.ainvoke([{"role": "user", "content": prompt}])
        raw = (res.content or "").strip()
        scores = _parse_scores(raw, expected_count=len(chunks))
        return scores
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker LLM call failed (%s), using original order.", exc)
        return list(range(len(chunks), 0, -1))  # descending dummy scores


def _parse_scores(raw: str, expected_count: int) -> List[float]:
    """
    Parse comma-separated scores from the LLM response.

    Falls back to equal descending scores if parsing fails or the count
    does not match.
    """
    # Extract all numbers (int or float) from the response
    numbers = re.findall(r"\d+(?:\.\d+)?", raw)
    if len(numbers) == expected_count:
        scores = [float(n) for n in numbers]
        # Clamp to [0, 10]
        return [max(0.0, min(10.0, s)) for s in scores]

    logger.warning(
        "Reranker score count mismatch: expected %d, got %d. Falling back.",
        expected_count,
        len(numbers),
    )
    return list(range(expected_count, 0, -1))


def _apply_scores(
    chunks: List[RetrievedChunk],
    scores: List[float],
) -> List[RetrievedChunk]:
    """Tag each chunk with its rerank_score and sort descending."""
    scored: List[Tuple[float, RetrievedChunk]] = []
    for chunk, score in zip(chunks, scores):
        chunk.metadata["rerank_score"] = score
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored]
