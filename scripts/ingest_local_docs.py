"""
ingest_local_docs.py
====================
Production-grade ingestion pipeline for local documents.

Pipeline flow
-------------
For each file in RAG/raw/:
  1. Load (PDF / DOCX / TXT / MD)
  2. Clean & normalize (text_cleaner)
  3. Smart adaptive chunk (chunker)
  4. Deduplicate against existing Chroma content_hashes
  5. Upsert new chunks with full metadata
  6. Log summary

Supported formats: .pdf, .docx, .txt, .md

Run
---
::

    python scripts/ingest_local_docs.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Set

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.chunker import chunk_documents
from app.rag.embedding_service import get_embedding_service
from app.rag.text_cleaner import infer_doc_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ingest_local_docs")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def iter_files(root: Path):
    for ext in SUPPORTED_EXTENSIONS:
        yield from root.rglob(f"*{ext}")


def load_document(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    # Ensure source metadata is always set
    for d in docs:
        d.metadata.setdefault("source", str(path))
    return docs


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def get_existing_hashes(chroma_path: Path, collection_name: str) -> Set[str]:
    """Fetch all content_hash values stored in the Chroma collection."""
    try:
        client = PersistentClient(path=str(chroma_path))
        col = client.get_collection(collection_name)
        result = col.get(include=["metadatas"])
        hashes: Set[str] = set()
        for meta in result.get("metadatas") or []:
            h = (meta or {}).get("content_hash")
            if h:
                hashes.add(h)
        return hashes
    except Exception:  # collection may not exist yet
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    raw_root = settings.rag_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    settings.chroma_local_docs_path.mkdir(parents=True, exist_ok=True)

    embedding_svc = get_embedding_service(settings)
    lc_embeddings = embedding_svc.get_langchain_embeddings()

    # Load existing content hashes to skip duplicates
    existing_hashes = get_existing_hashes(
        settings.chroma_local_docs_path, "local_docs"
    )
    logger.info("Existing Chroma hashes: %d", len(existing_hashes))

    files = list(iter_files(raw_root))
    if not files:
        logger.warning("No documents found in %s — nothing to ingest.", raw_root)
        return
    logger.info("Found %d files to process.", len(files))

    all_chunks = []
    skipped_dups = 0

    for file_path in files:
        logger.info("Processing: %s", file_path.name)
        try:
            raw_docs = load_document(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", file_path, exc)
            continue

        doc_type = infer_doc_type(str(file_path))

        chunks = chunk_documents(
            docs=raw_docs,
            doc_type=doc_type,
            source_type="local",
            min_chunk_size=settings.chunk_min_size,
            target_chunk_size=settings.chunk_target_size,
            max_chunk_size=settings.chunk_max_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Dedup against already-stored chunks
        new_chunks: List = []
        for chunk in chunks:
            h = chunk.metadata.get("content_hash", "")
            if h and h in existing_hashes:
                skipped_dups += 1
                continue
            existing_hashes.add(h)
            new_chunks.append(chunk)

        logger.info(
            "  %s → %d raw chunks, %d new, %d skipped (dup)",
            file_path.name,
            len(chunks),
            len(new_chunks),
            len(chunks) - len(new_chunks),
        )
        all_chunks.extend(new_chunks)

    if not all_chunks:
        logger.info(
            "All %d chunks already indexed (or no new content). Nothing to upsert.",
            skipped_dups,
        )
        return

    logger.info("Upserting %d new chunks into Chroma …", len(all_chunks))

    Chroma.from_documents(
        documents=all_chunks,
        embedding=lc_embeddings,
        persist_directory=str(settings.chroma_local_docs_path),
        collection_name="local_docs",
    )

    logger.info(
        "✓ Chroma 'local_docs' updated — %d new chunks added, %d duplicates skipped.",
        len(all_chunks),
        skipped_dups,
    )


if __name__ == "__main__":
    main()
