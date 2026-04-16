"""
tracing.py
==========
LangSmith observability integration for the TEKNOFEST RAG pipeline.

How it works
------------
LangChain + LangGraph auto-trace ALL calls (LLM, vectorstore, chains) when
the standard ``LANGCHAIN_*`` environment variables are set.  This module:

1. Reads those variables from ``Settings`` and exports them as ``os.environ``
   entries so the LangChain internals pick them up — even if they were
   loaded from ``.env`` after the process started.

2. Provides ``@traceable`` wrappers (``embedding_step``, ``retrieval_step``,
   ``rerank_step``, ``context_builder``) for non-LangChain functions so they
   appear as first-class spans in the LangSmith trace graph.

3. Provides ``get_run_metadata()`` — a helper to attach structured metadata
   (retrieved chunks, rerank scores, prompt preview) to the current run.

4. Guarantees graceful fallback: if LangSmith is unavailable or the API key
   is missing, every decorated function still executes normally.

Usage (idempotent — safe to call multiple times)
-----
::

    from app.tracing import init_langsmith
    init_langsmith(settings)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard — langsmith is optional at runtime
# ---------------------------------------------------------------------------
try:
    from langsmith import traceable as _traceable
    from langsmith import Client as _LangSmithClient
    _LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGSMITH_AVAILABLE = False
    logger.warning("langsmith package not found — tracing disabled.")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

_initialized = False


def init_langsmith(settings) -> bool:
    """
    Configure LangSmith tracing from ``settings``.

    Sets the required ``LANGCHAIN_*`` environment variables so that
    LangChain / LangGraph can auto-detect them.

    Parameters
    ----------
    settings : Settings
        Loaded application settings.

    Returns
    -------
    bool
        ``True`` if tracing was enabled, ``False`` otherwise.
    """
    global _initialized

    if not _LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not available — skipping tracing init.")
        return False

    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2 != true).")
        return False

    if not settings.langsmith_api_key:
        logger.warning(
            "LangSmith tracing requested but LANGCHAIN_API_KEY is not set — "
            "tracing will be skipped."
        )
        return False

    try:
        # Export vars so LangChain internals pick them up.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

        if not _initialized:
            logger.info(
                "LangSmith tracing ENABLED — project=%r  endpoint=%s",
                settings.langsmith_project,
                settings.langsmith_endpoint,
            )
            _initialized = True

        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("LangSmith init failed (non-fatal): %s", exc)
        return False


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is currently active."""
    return (
        _LANGSMITH_AVAILABLE
        and os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.environ.get("LANGCHAIN_API_KEY"))
    )


# ---------------------------------------------------------------------------
# @traceable decorator factory
# ---------------------------------------------------------------------------

def _safe_traceable(name: str, run_type: str = "chain"):
    """
    Return a ``@langsmith.traceable`` decorator when tracing is available,
    or a pass-through identity decorator when it is not.

    This guarantees the decorated functions always work regardless of
    whether LangSmith is configured.
    """
    if _LANGSMITH_AVAILABLE:
        return _traceable(name=name, run_type=run_type)

    # Identity decorator fallback
    def _identity(fn):
        return fn
    return _identity


# ---------------------------------------------------------------------------
# Named span decorators (re-exported for use in pipeline modules)
# ---------------------------------------------------------------------------

#: Wraps embed_documents / embed_query calls
embedding_step = _safe_traceable("embedding_step", run_type="embedding")

#: Wraps retrieve_from_vectorstore calls
retrieval_step = _safe_traceable("retrieval_step", run_type="retriever")

#: Wraps rerank_chunks
rerank_step = _safe_traceable("rerank_step", run_type="chain")

#: Wraps build_context
context_builder_step = _safe_traceable("context_builder", run_type="chain")

#: Wraps full answer generation
llm_generation_step = _safe_traceable("llm_generation", run_type="llm")


# ---------------------------------------------------------------------------
# Run metadata helper
# ---------------------------------------------------------------------------

def get_run_metadata(
    retrieved_chunks,
    selected_chunks=None,
    prompt_preview: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a structured metadata dict to attach to a LangSmith run.

    Parameters
    ----------
    retrieved_chunks : List[RetrievedChunk]
        All candidates from the vector store.
    selected_chunks : List[RetrievedChunk] | None
        Final context chunks after reranking / compression.
    prompt_preview : str | None
        First 500 chars of the final prompt sent to the LLM.
    extra : dict | None
        Any additional key-value pairs to include.

    Returns
    -------
    dict
        Flat JSON-serialisable dict suitable for ``langsmith_extra``.
    """
    meta: Dict[str, Any] = {
        "retrieved_count": len(retrieved_chunks) if retrieved_chunks else 0,
        "selected_count": len(selected_chunks) if selected_chunks else 0,
        "retrieved_sources": [
            {
                "source": ch.metadata.get("source") or ch.metadata.get("url"),
                "doc_type": ch.metadata.get("document_type", ch.source_type),
                "score": ch.score,
                "rerank_score": ch.metadata.get("rerank_score"),
                "section": ch.metadata.get("section_title"),
                "page": ch.metadata.get("page_number"),
            }
            for ch in (retrieved_chunks or [])
        ],
        "rerank_scores": [
            ch.metadata.get("rerank_score") for ch in (selected_chunks or [])
        ],
    }
    if prompt_preview:
        meta["prompt_preview"] = prompt_preview[:500]
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# LangSmith client (for dataset / evaluation operations)
# ---------------------------------------------------------------------------

def get_langsmith_client() -> Optional[Any]:
    """
    Return an authenticated ``langsmith.Client`` instance, or ``None`` if
    tracing is not configured.
    """
    if not _LANGSMITH_AVAILABLE or not is_tracing_enabled():
        return None
    try:
        return _LangSmithClient(
            api_url=os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            api_key=os.environ.get("LANGCHAIN_API_KEY"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build LangSmith client: %s", exc)
        return None
