"""
evaluator.py
============
Evaluation and logging layer for the RAG pipeline.

Features
--------
1. **JSONL event logging** — every query logs: query, retrieved chunks,
   reranked chunks, selected context, generated answer, route, latency.
2. **Recall@K** — did any ground-truth keyword appear in the top-K results?
3. **Context Relevance Scoring** — LLM-graded relevance (0-1) per chunk.
4. **Failure tagging** — ``bad_retrieval``, ``hallucination``, ``partial_answer``.
5. **Ground-truth evaluation** — run against a JSON dataset and produce a
   summary report.

Log format (JSONL)
------------------
One JSON object per line::

    {
      "ts": "2026-04-15T21:00:00Z",
      "query": "...",
      "route": "local",
      "retrieved_count": 10,
      "selected_count": 5,
      "answer_preview": "...",
      "latency_ms": 1234,
      "failure_tags": [],
      "hallucination_status": "safe",
      "retrieved_metadata": [...],
      "reranked_scores": [...]
    }
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.retrievers import RetrievedChunk

logger = logging.getLogger(__name__)

FailureTag = str  # "bad_retrieval" | "hallucination" | "partial_answer"


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------


def log_retrieval_event(
    *,
    log_path: Path,
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    selected_chunks: List[RetrievedChunk],
    answer: str,
    route: str,
    hallucination_status: str = "unknown",
    failure_tags: Optional[List[FailureTag]] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """
    Append one retrieval event to the JSONL log file.

    Parameters are intentionally keyword-only to avoid positional confusion.
    The log file is created (including parent directories) if it doesn't exist.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    event: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "route": route,
        "retrieved_count": len(retrieved_chunks),
        "selected_count": len(selected_chunks),
        "answer_preview": answer[:300] if answer else "",
        "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        "failure_tags": failure_tags or [],
        "hallucination_status": hallucination_status,
        "retrieved_metadata": [
            {
                "source": ch.metadata.get("source") or ch.metadata.get("url"),
                "doc_type": ch.metadata.get("document_type", ch.source_type),
                "page": ch.metadata.get("page_number"),
                "section": ch.metadata.get("section_title"),
                "score": ch.score,
                "rerank_score": ch.metadata.get("rerank_score"),
                "content_hash": ch.metadata.get("content_hash", "")[:12],
            }
            for ch in retrieved_chunks
        ],
        "reranked_scores": [
            ch.metadata.get("rerank_score") for ch in selected_chunks
        ],
    }

    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to write eval log: %s", exc)


# ---------------------------------------------------------------------------
# Ground-truth evaluation
# ---------------------------------------------------------------------------


def load_eval_dataset(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON evaluation dataset.

    Expected format::

        [
          {
            "query": "TEKNOFEST ne zaman başladı?",
            "expected_keywords": ["2018", "ilk", "başladı"],
            "failure_tags": []
          },
          ...
        ]
    """
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("eval_dataset.json must be a JSON array")
    return data


def recall_at_k(
    retrieved_chunks: List[RetrievedChunk],
    expected_keywords: List[str],
    k: int,
) -> float:
    """
    Recall@K: fraction of expected_keywords found in top-k chunk texts.

    Returns a float in [0.0, 1.0].
    """
    if not expected_keywords:
        return 1.0  # nothing to recall
    top_k = retrieved_chunks[:k]
    combined_text = " ".join(ch.content.lower() for ch in top_k)
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords)


def tag_failures(
    retrieved_chunks: List[RetrievedChunk],
    answer: str,
    hallucination_status: str,
    recall: float,
    recall_threshold: float = 0.5,
) -> List[FailureTag]:
    """
    Automatically tag known failure modes.

    Tags assigned
    -------------
    bad_retrieval   : recall < recall_threshold
    hallucination   : hallucination_status == "suspicious"
    partial_answer  : answer is very short (< 50 chars)
    """
    tags: List[FailureTag] = []
    if recall < recall_threshold:
        tags.append("bad_retrieval")
    if hallucination_status == "suspicious":
        tags.append("hallucination")
    if len(answer.strip()) < 50:
        tags.append("partial_answer")
    return tags


# ---------------------------------------------------------------------------
# Timer context manager (convenience)
# ---------------------------------------------------------------------------


class Timer:
    """Simple wall-clock timer for measuring latency."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
