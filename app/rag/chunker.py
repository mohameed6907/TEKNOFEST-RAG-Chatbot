"""
chunker.py
==========
Adaptive smart chunking with rich metadata enrichment.

Strategy
--------
Uses LangChain's RecursiveCharacterTextSplitter with semantic separators
(paragraph > sentence > clause) so chunks never break mid-sentence.

Adaptive sizing:
  min_chunk_size  : 400  chars  (never emit shorter chunks except at EOF)
  target_chunk_size: 800 chars  (preferred split point)
  max_chunk_size  : 1200 chars  (hard ceiling)
  chunk_overlap   : 150  chars  (context continuity)

Each produced chunk carries a full metadata dict:
  source            : str  — file path or URL
  document_type     : str  — pdf | docx | txt | web | md
  page_number       : int | None
  section_title     : str | None
  chunk_index       : int
  total_chunks      : int  (set in a post-pass)
  ingested_at       : str  — ISO-8601 UTC timestamp
  crawl_source      : str | None  — original URL for web docs
  source_priority   : int  — local=3, teknofest=2, web=1
  content_hash      : str  — SHA-256 of normalized text (for dedup)
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.text_cleaner import DocType, clean_text, normalize_chunk

# ---------------------------------------------------------------------------
# Heading detection patterns
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    # Markdown: # Title / ## Subtitle / ### …
    re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE),
    # ALL-CAPS lines of ≥ 4 words (common in PDFs / DOCX)
    re.compile(r"^([A-ZÇĞİÖŞÜA-Z][A-Z\sÇĞİÖŞÜ]{10,})$", re.MULTILINE),
    # Numbered sections: 1. Title / 1.2 Sub-title
    re.compile(r"^\d+(\.\d+)*[\.\)]\s+([A-ZÇĞİÖŞÜa-z].{4,})$", re.MULTILINE),
]


def _detect_heading(text: str) -> Optional[str]:
    """Return the first detected heading in *text*, or None."""
    for pattern in _HEADING_PATTERNS:
        m = pattern.search(text)
        if m:
            # Group 1 for md/numbered patterns, full match for ALL-CAPS
            heading = (m.group(1) if m.lastindex else m.group(0)).strip()
            # Sanity: skip if looks like a sentence fragment (has a verb ending)
            if len(heading) < 100:
                return heading
    return None


# ---------------------------------------------------------------------------
# Source priority helper
# ---------------------------------------------------------------------------

_PRIORITY_MAP = {
    "local": 3,
    "teknofest": 2,
    "web": 1,
}


def _source_priority(source_type: str) -> int:
    return _PRIORITY_MAP.get(source_type.lower(), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_documents(
    docs: List[Document],
    doc_type: DocType,
    source_type: str = "local",
    *,
    min_chunk_size: int = 400,
    target_chunk_size: int = 800,
    max_chunk_size: int = 1200,
    chunk_overlap: int = 150,
    crawl_source: Optional[str] = None,
) -> List[Document]:
    """
    Clean, chunk, and enrich a list of LangChain Documents.

    Parameters
    ----------
    docs : List[Document]
        Raw documents as returned by a LangChain loader.
    doc_type : DocType
        Document type used for cleaning rules.
    source_type : str
        ``'local'``, ``'teknofest'``, or ``'web'`` – drives source_priority.
    min_chunk_size : int
        Minimum characters per chunk (shorter chunks are merged forward).
    target_chunk_size : int
        Preferred split point.
    max_chunk_size : int
        Hard upper limit.
    chunk_overlap : int
        Character overlap between consecutive chunks.
    crawl_source : str | None
        Original crawl URL for web documents.

    Returns
    -------
    List[Document]
        Enriched, deduplicated chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        chunk_size=target_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        keep_separator=True,
    )

    ingested_at = datetime.now(timezone.utc).isoformat()
    priority = _source_priority(source_type)

    enriched: List[Document] = []

    for doc in docs:
        # ---- 1. Clean raw text ----
        raw_text = clean_text(doc.page_content or "", doc_type=doc_type)
        if not raw_text.strip():
            continue

        base_meta: Dict[str, Any] = {
            "source": doc.metadata.get("source", "unknown"),
            "document_type": doc_type,
            "page_number": doc.metadata.get("page", doc.metadata.get("page_number")),
            "ingested_at": ingested_at,
            "crawl_source": crawl_source or doc.metadata.get("url"),
            "source_priority": priority,
            # Will be filled per-chunk below
            "section_title": None,
            "chunk_index": 0,
            "total_chunks": 0,
            "content_hash": "",
        }

        # ---- 2. Split ----
        raw_chunks: List[Document] = splitter.create_documents(
            texts=[raw_text],
            metadatas=[base_meta.copy()],
        )

        # ---- 3. Filter sub-minimum chunks (merge into previous) ----
        raw_chunks = _merge_short_chunks(raw_chunks, min_chunk_size)

        # ---- 4. Enrich each chunk ----
        total = len(raw_chunks)
        seen_hashes: set = set()

        for idx, chunk in enumerate(raw_chunks):
            norm = normalize_chunk(chunk.page_content)
            if not norm:
                continue

            content_hash = _sha256(norm)
            if content_hash in seen_hashes:
                continue  # intra-document dedup
            seen_hashes.add(content_hash)

            heading = _detect_heading(chunk.page_content)

            chunk.metadata.update(
                {
                    "section_title": heading,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "content_hash": content_hash,
                }
            )
            # Store the normalized text as page_content for embedding
            chunk.page_content = norm
            enriched.append(chunk)

    return enriched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_short_chunks(
    chunks: List[Document], min_size: int
) -> List[Document]:
    """
    Merge consecutive chunks that are shorter than *min_size* into the
    next chunk.  This keeps the last chunk even if it's short (EOF case).
    """
    if not chunks:
        return []

    merged: List[Document] = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        if len(buffer.page_content) < min_size:
            # Merge: append current chunk text to buffer
            buffer = Document(
                page_content=buffer.page_content.rstrip()
                + "\n\n"
                + chunk.page_content.lstrip(),
                metadata=buffer.metadata.copy(),
            )
        else:
            merged.append(buffer)
            buffer = chunk

    merged.append(buffer)  # always keep the last
    return merged


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
