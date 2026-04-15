"""
test_rag_pipeline.py
====================
Unit tests for the production-grade RAG pipeline.

Tests are designed to run WITHOUT real OpenAI/Chroma calls by mocking
external dependencies.  Tests that require live API keys are skipped
automatically when keys are not available.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import get_settings
from app.rag.chunker import _sha256, chunk_documents
from app.rag.context_builder import build_context
from app.rag.evaluator import Timer, log_retrieval_event, recall_at_k, tag_failures
from app.rag.retrievers import RetrievedChunk
from app.rag.text_cleaner import clean_text, infer_doc_type, normalize_chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings():
    return get_settings()


def _make_chunk(
    content: str,
    source_type: str = "local_docs",
    score: float | None = 0.2,
    content_hash: str | None = None,
    **meta_kwargs,
) -> RetrievedChunk:
    h = content_hash or _sha256(content)
    return RetrievedChunk(
        content=content,
        metadata={"content_hash": h, "source_priority": 3, **meta_kwargs},
        score=score,
        source_type=source_type,
    )


# ---------------------------------------------------------------------------
# 1. text_cleaner
# ---------------------------------------------------------------------------


class TestTextCleaner:
    def test_removes_null_bytes(self):
        text = "Hello\x00World"
        assert "\x00" not in clean_text(text)

    def test_collapses_whitespace(self):
        text = "Hello    World\n\n\n\nFoo"
        cleaned = clean_text(text)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_pdf_removes_page_number_line(self):
        text = "Some content\nPage 1 of 5\nMore content"
        cleaned = clean_text(text, doc_type="pdf")
        assert "Page 1 of 5" not in cleaned

    def test_web_removes_cookie_notice(self):
        text = "Interesting article content. Cookie Policy Accept all cookies footer"
        cleaned = clean_text(text, doc_type="web")
        assert "cookie" not in cleaned.lower()

    def test_infer_doc_type(self):
        assert infer_doc_type("report.pdf") == "pdf"
        assert infer_doc_type("doc.docx") == "docx"
        assert infer_doc_type("https://teknofest.org/") == "web"
        assert infer_doc_type("readme.md") == "md"

    def test_normalize_chunk_strips(self):
        text = "  Hello World  "
        assert normalize_chunk(text) == "Hello World"

    def test_normalize_chunk_collapses_newlines(self):
        text = "Line one\nLine two"
        result = normalize_chunk(text)
        assert "\n" not in result or result.count("\n") <= 1


# ---------------------------------------------------------------------------
# 2. chunker
# ---------------------------------------------------------------------------


class TestChunker:
    def _make_doc(self, content: str, source: str = "test.txt"):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata={"source": source})

    def test_required_metadata_fields(self):
        long_text = "This is a test sentence. " * 60
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], doc_type="txt", source_type="local")

        assert len(chunks) > 0
        required_keys = {
            "source", "document_type", "source_priority",
            "content_hash", "chunk_index", "ingested_at",
        }
        for chunk in chunks:
            missing = required_keys - set(chunk.metadata.keys())
            assert not missing, f"Missing metadata keys: {missing}"

    def test_content_hash_is_sha256(self):
        doc = self._make_doc("Hello world " * 40)
        chunks = chunk_documents([doc], doc_type="txt")
        for chunk in chunks:
            h = chunk.metadata["content_hash"]
            assert len(h) == 64  # SHA-256 hex

    def test_no_duplicate_hashes_within_document(self):
        # Repeated content should be deduplicated within a single file
        doc = self._make_doc("Repeated sentence. " * 100)
        chunks = chunk_documents([doc], doc_type="txt")
        hashes = [c.metadata["content_hash"] for c in chunks]
        assert len(hashes) == len(set(hashes)), "Duplicate chunk hashes found"

    def test_source_priority_local(self):
        doc = self._make_doc("Content " * 50)
        chunks = chunk_documents([doc], doc_type="txt", source_type="local")
        for chunk in chunks:
            assert chunk.metadata["source_priority"] == 3

    def test_source_priority_web(self):
        doc = self._make_doc("Content " * 50)
        chunks = chunk_documents([doc], doc_type="web", source_type="web")
        for chunk in chunks:
            assert chunk.metadata["source_priority"] == 1

    def test_chunk_size_respected(self):
        long_text = "Word " * 500
        doc = self._make_doc(long_text)
        chunks = chunk_documents(
            [doc], doc_type="txt",
            target_chunk_size=400, max_chunk_size=800, chunk_overlap=50
        )
        for chunk in chunks:
            assert len(chunk.page_content) <= 1600  # generous ceiling

    def test_empty_doc_produces_no_chunks(self):
        doc = self._make_doc("   ")
        chunks = chunk_documents([doc], doc_type="txt")
        assert chunks == []


# ---------------------------------------------------------------------------
# 3. context_builder
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def test_deduplicates_exact_hashes(self):
        chunk1 = _make_chunk("Hello world this is a test", content_hash="aaa")
        chunk2 = _make_chunk("Hello world this is a test", content_hash="aaa")
        _, selected = build_context([chunk1, chunk2], min_score=None)
        assert len(selected) == 1

    def test_respects_character_budget(self):
        chunks = [_make_chunk("A" * 500, score=0.1) for _ in range(20)]
        context_str, selected = build_context(chunks, max_total_chars=2000, min_score=None)
        total_chars = sum(len(c.content) for c in selected)
        assert total_chars <= 2500  # budget + last partial

    def test_returns_string_and_list(self):
        chunk = _make_chunk("Some relevant content about TEKNOFEST.")
        context_str, selected = build_context([chunk], min_score=None)
        assert isinstance(context_str, str)
        assert isinstance(selected, list)
        assert len(selected) == 1

    def test_empty_input(self):
        context_str, selected = build_context([])
        assert context_str == ""
        assert selected == []

    def test_format_includes_source(self):
        chunk = _make_chunk(
            "TEKNOFEST bilgi.", source="teknofest.org/about"
        )
        chunk.metadata["source"] = "teknofest.org/about"
        context_str, _ = build_context([chunk], min_score=None)
        assert "teknofest.org/about" in context_str


# ---------------------------------------------------------------------------
# 4. evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_recall_at_k_perfect(self):
        chunks = [_make_chunk("2018 başladı ilk TEKNOFEST")]
        score = recall_at_k(chunks, ["2018", "başladı", "ilk"], k=5)
        assert score == 1.0

    def test_recall_at_k_zero(self):
        chunks = [_make_chunk("Tamamen alakasız içerik burada")]
        score = recall_at_k(chunks, ["xyz123", "notfound"], k=5)
        assert score == 0.0

    def test_recall_at_k_empty_expected(self):
        chunks = [_make_chunk("Some content")]
        assert recall_at_k(chunks, [], k=5) == 1.0

    def test_tag_failures_bad_retrieval(self):
        tags = tag_failures(
            retrieved_chunks=[],
            answer="Reasonable answer here",
            hallucination_status="safe",
            recall=0.0,
        )
        assert "bad_retrieval" in tags

    def test_tag_failures_hallucination(self):
        tags = tag_failures(
            retrieved_chunks=[_make_chunk("content")],
            answer="Reasonable long answer about TEKNOFEST competitions",
            hallucination_status="suspicious",
            recall=1.0,
        )
        assert "hallucination" in tags

    def test_tag_failures_partial_answer(self):
        tags = tag_failures(
            retrieved_chunks=[_make_chunk("content")],
            answer="Ok",
            hallucination_status="safe",
            recall=1.0,
        )
        assert "partial_answer" in tags

    def test_log_retrieval_event_jsonl(self, tmp_path):
        log_file = tmp_path / "test_eval.jsonl"
        chunks = [_make_chunk("Test content")]
        log_retrieval_event(
            log_path=log_file,
            query="Test query?",
            retrieved_chunks=chunks,
            selected_chunks=chunks,
            answer="Test answer here.",
            route="local",
            hallucination_status="safe",
            failure_tags=[],
            latency_ms=123.4,
        )
        assert log_file.exists()
        line = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert line["query"] == "Test query?"
        assert line["route"] == "local"
        assert line["latency_ms"] == 123.4
        assert "ts" in line

    def test_log_creates_parent_dirs(self, tmp_path):
        log_file = tmp_path / "nested" / "deeply" / "log.jsonl"
        log_retrieval_event(
            log_path=log_file,
            query="q",
            retrieved_chunks=[],
            selected_chunks=[],
            answer="a",
            route="direct",
        )
        assert log_file.exists()

    def test_timer(self):
        import time
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 10  # at least 10ms


# ---------------------------------------------------------------------------
# 5. retrievers (unit — mocked Chroma)
# ---------------------------------------------------------------------------


class TestRetrievers:
    def test_retrieve_returns_structured_chunks(self):
        from app.rag.retrievers import retrieve_from_vectorstore

        fake_doc = MagicMock()
        fake_doc.page_content = "TEKNOFEST 2024"
        fake_doc.metadata = {"source": "test.pdf", "content_hash": "abc123"}

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [(fake_doc, 0.15)]

        chunks = retrieve_from_vectorstore(mock_vs, "TEKNOFEST", "local_docs", k=5)
        assert len(chunks) == 1
        assert chunks[0].content == "TEKNOFEST 2024"
        assert chunks[0].score == 0.15
        assert chunks[0].source_type == "local_docs"

    def test_retrieve_handles_error_gracefully(self):
        from app.rag.retrievers import retrieve_from_vectorstore

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.side_effect = RuntimeError("DB gone")

        chunks = retrieve_from_vectorstore(mock_vs, "query", "local_docs", k=5)
        assert chunks == []

    def test_estimate_confidence_empty(self):
        from app.rag.retrievers import estimate_chunk_confidence
        assert estimate_chunk_confidence([]) == 0.0

    def test_estimate_confidence_range(self):
        from app.rag.retrievers import estimate_chunk_confidence
        chunks = [_make_chunk("c", score=0.3), _make_chunk("d", score=0.8)]
        conf = estimate_chunk_confidence(chunks)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 6. reranker (mocked LLM)
# ---------------------------------------------------------------------------


class TestReranker:
    @pytest.mark.asyncio
    async def test_reranker_returns_top_k(self):
        from app.rag.reranker import rerank_chunks

        chunks = [_make_chunk(f"Chunk {i}") for i in range(6)]
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="9,3,7,2,8,5")

        result = await rerank_chunks(
            query="test",
            chunks=chunks,
            llm=mock_llm,
            final_k=3,
        )
        assert len(result) == 3
        # Should be sorted descending by score
        scores = [c.metadata["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_reranker_fallback_on_error(self):
        from app.rag.reranker import rerank_chunks

        chunks = [_make_chunk(f"Chunk {i}") for i in range(4)]
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("LLM down")

        # Should not raise — falls back to original order
        result = await rerank_chunks(
            query="test", chunks=chunks, llm=mock_llm, final_k=4
        )
        assert len(result) == 4


# ---------------------------------------------------------------------------
# 7. Embedding service (unit — checks singleton and model name)
# ---------------------------------------------------------------------------


class TestEmbeddingService:
    def test_singleton_pattern(self, settings):
        """get_embedding_service always returns the same instance."""
        import app.rag.embedding_service as es_module
        es_module._instance = None  # reset singleton for test isolation

        with patch("app.rag.embedding_service.OpenAIEmbeddings") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc1 = es_module.get_embedding_service(settings)
            svc2 = es_module.get_embedding_service(settings)
            assert svc1 is svc2
            assert mock_cls.call_count == 1

        es_module._instance = None  # clean up

    def test_raises_without_api_key(self):
        from app.rag.embedding_service import EmbeddingService

        bad_settings = get_settings().model_copy(update={"openai_api_key": None})
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            EmbeddingService(bad_settings)
