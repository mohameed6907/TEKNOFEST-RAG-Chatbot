"""
context_builder.py
==================
Context compression and construction before the final LLM call.

Responsibilities
----------------
1. **Deduplication** — remove chunks whose content_hash is identical or
   whose text overlap with an already-selected chunk exceeds a threshold.
2. **Relevance trimming** — drop chunks with a similarity score below
   ``min_score`` (distance-based) or a rerank_score below ``min_rerank``.
3. **Merging** — merge adjacent chunks from the same source/page when their
   combined length fits within ``max_total_chars``.
4. **Context assembly** — format surviving chunks into a numbered, labeled
   context block ready for the LLM prompt.

All steps are configurable and have sensible defaults.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.rag.retrievers import RetrievedChunk

try:
    from langsmith import traceable as _ls_traceable
    _context_builder_step = _ls_traceable(name="context_builder", run_type="chain")
except ImportError:  # pragma: no cover
    def _context_builder_step(fn):  # type: ignore[misc]
        return fn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TOTAL_CHARS = 6_000   # ~1500 tokens at avg 4 chars/token
_DEFAULT_MIN_SCORE = 0.35          # Chroma distance threshold (lower = closer)
_DEFAULT_MIN_RERANK = 2.0          # minimum rerank_score to keep
_OVERLAP_THRESHOLD = 0.80          # Jaccard word-overlap to flag near-duplicates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@_context_builder_step
def build_context(
    chunks: List[RetrievedChunk],
    *,
    max_total_chars: int = _DEFAULT_MAX_TOTAL_CHARS,
    min_score: Optional[float] = _DEFAULT_MIN_SCORE,
    min_rerank: Optional[float] = None,
) -> tuple[str, List[RetrievedChunk]]:
    """
    Compress and format retrieved chunks into a final context string.

    Parameters
    ----------
    chunks : List[RetrievedChunk]
        Ordered list (best-first) of retrieved / reranked chunks.
    max_total_chars : int
        Hard character budget for the assembled context.
    min_score : float | None
        If set, drop chunks whose distance score exceeds this (i.e. too far).
        Set to ``None`` to skip score filtering.
    min_rerank : float | None
        If set, drop chunks whose ``rerank_score`` metadata is below this.
        Set to ``None`` to skip rerank score filtering.

    Returns
    -------
    (context_str, selected_chunks)
        ``context_str`` — formatted block to inject into the LLM prompt.
        ``selected_chunks`` — the subset that was used.
    """
    if not chunks:
        return "", []

    # Step 1: score-based filtering
    filtered = _filter_by_score(chunks, min_score=min_score, min_rerank=min_rerank)

    # Step 2: deduplication (exact hash + near-duplicate by word overlap)
    deduped = _deduplicate(filtered)

    # Step 3: assemble within budget
    selected = _select_within_budget(deduped, max_total_chars)

    # Step 4: format
    context_str = _format_context(selected)

    logger.debug(
        "Context builder: %d → scored %d → deduped %d → selected %d chunks",
        len(chunks),
        len(filtered),
        len(deduped),
        len(selected),
    )

    return context_str, selected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filter_by_score(
    chunks: List[RetrievedChunk],
    min_score: Optional[float],
    min_rerank: Optional[float],
) -> List[RetrievedChunk]:
    result = []
    for ch in chunks:
        # Distance-based Chroma score (lower is better)
        if min_score is not None and ch.score is not None:
            if ch.score > min_score:
                logger.debug("Dropping chunk (score %.3f > %.3f)", ch.score, min_score)
                continue
        # LLM rerank score (higher is better)
        if min_rerank is not None:
            rs = ch.metadata.get("rerank_score")
            if rs is not None and float(rs) < min_rerank:
                logger.debug("Dropping chunk (rerank_score %.1f < %.1f)", rs, min_rerank)
                continue
        result.append(ch)
    return result


def _deduplicate(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Remove exact duplicates (by content_hash) and near-duplicates."""
    seen_hashes: set = set()
    seen_words: List[set] = []
    result = []

    for ch in chunks:
        # Exact dedup via hash
        h = ch.metadata.get("content_hash", "")
        if h and h in seen_hashes:
            logger.debug("Dropping exact duplicate (hash=%s…)", h[:8])
            continue
        if h:
            seen_hashes.add(h)

        # Near-duplicate via Jaccard word overlap
        words = set(ch.content.lower().split())
        for seen in seen_words:
            if len(words) == 0 or len(seen) == 0:
                continue
            intersection = words & seen
            union = words | seen
            jaccard = len(intersection) / len(union)
            if jaccard >= _OVERLAP_THRESHOLD:
                logger.debug("Dropping near-duplicate (jaccard=%.2f)", jaccard)
                break
        else:
            seen_words.append(words)
            result.append(ch)

    return result


def _select_within_budget(
    chunks: List[RetrievedChunk], budget: int
) -> List[RetrievedChunk]:
    """Take chunks greedily until the character budget is reached."""
    selected = []
    used = 0
    for ch in chunks:
        length = len(ch.content)
        if used + length > budget and selected:
            # See if a shorter chunk fits
            continue
        selected.append(ch)
        used += length
        if used >= budget:
            break
    return selected


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """Format chunks into a numbered, source-labeled context block."""
    lines = []
    for idx, ch in enumerate(chunks, start=1):
        meta = ch.metadata
        src = (
            meta.get("source")
            or meta.get("crawl_source")
            or meta.get("url")
            or "unknown"
        )
        doc_type = meta.get("document_type", ch.source_type)
        page = meta.get("page_number")
        section = meta.get("section_title")

        label_parts = [f"[{idx}]", f"({doc_type})"]
        if section:
            label_parts.append(f'"{section}"')
        if page is not None:
            label_parts.append(f"p.{page}")
        label_parts.append(src)

        lines.append(" ".join(label_parts))
        lines.append(ch.content)
        lines.append("")  # blank line separator

    return "\n".join(lines).strip()
