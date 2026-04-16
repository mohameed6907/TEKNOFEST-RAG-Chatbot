"""
embedding_service.py
====================
Single source of truth for all embedding operations.

Rules
-----
- ONLY OpenAI embeddings (text-embedding-3-small / text-embedding-3-large)
- Used by BOTH ingestion scripts and the retrieval layer
- No embedding logic should exist outside this module
- Supports batching and error handling with exponential backoff
"""
from __future__ import annotations

import logging
import time
from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import Settings

try:
    from langsmith import traceable as _traceable
    _embedding_step = _traceable(name="embedding_step", run_type="embedding")
except ImportError:  # pragma: no cover
    def _embedding_step(fn):  # type: ignore[misc]
        return fn

logger = logging.getLogger(__name__)

# OpenAI embedding API limit (max texts per request)
_OPENAI_BATCH_LIMIT = 2048


class EmbeddingService:
    """
    Wraps OpenAI embeddings with batching and retry logic.

    Usage
    -----
    ::

        from app.rag.embedding_service import get_embedding_service
        svc = get_embedding_service(settings)
        vectors = svc.embed_documents(["Hello", "World"])
        query_vec = svc.embed_query("What is TEKNOFEST?")
        lc_embeddings = svc.get_langchain_embeddings()   # pass to Chroma
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the embedding service. "
                "Set it in your .env file."
            )
        self._model = settings.embedding_model_name
        self._api_key = settings.openai_api_key
        self._lc_embeddings = OpenAIEmbeddings(
            model=self._model,
            openai_api_key=self._api_key,
        )
        logger.info("EmbeddingService initialised with model=%s", self._model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_langchain_embeddings(self) -> OpenAIEmbeddings:
        """Return the LangChain-compatible embeddings object (e.g. for Chroma)."""
        return self._lc_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents with automatic batching.

        Parameters
        ----------
        texts : List[str]
            Texts to embed. Empty strings are replaced with a single space
            to avoid API errors.

        Returns
        -------
        List[List[float]]
            One vector per input text, same order.
        """
        if not texts:
            return []
        sanitized = [t.strip() or " " for t in texts]
        vectors: List[List[float]] = []
        for batch in self._batch(sanitized, _OPENAI_BATCH_LIMIT):
            vectors.extend(self._embed_with_retry(batch))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string — traced as embedding_step in LangSmith."""
        return _traced_embed_query(self._lc_embeddings, text.strip() or " ")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch(items: List[str], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _embed_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> List[List[float]]:
        """Call OpenAI with exponential backoff on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return _traced_embed_documents(self._lc_embeddings, texts)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = base_delay * (2**attempt)
                logger.warning(
                    "Embedding attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Embedding failed after {max_retries} attempts"
        ) from last_exc


# ---------------------------------------------------------------------------
# Module-level singleton factory
# ---------------------------------------------------------------------------

_instance: EmbeddingService | None = None


def get_embedding_service(settings: Settings) -> EmbeddingService:
    """
    Return a module-level singleton EmbeddingService.

    Calling this multiple times with the same settings object returns
    the same instance (no re-initialisation).
    """
    global _instance
    if _instance is None:
        _instance = EmbeddingService(settings)
    return _instance


# ---------------------------------------------------------------------------
# Module-level @traceable wrappers (LangSmith span: "embedding_step")
# ---------------------------------------------------------------------------
# These are module-level functions (not instance methods) so the @traceable
# decorator can wrap them normally.  EmbeddingService delegates to them.

@_embedding_step
def _traced_embed_documents(
    lc_embeddings, texts: List[str]
) -> List[List[float]]:
    """Traced wrapper for batch document embedding."""
    return lc_embeddings.embed_documents(texts)


@_embedding_step
def _traced_embed_query(lc_embeddings, text: str) -> List[float]:
    """Traced wrapper for single query embedding."""
    return lc_embeddings.embed_query(text)
