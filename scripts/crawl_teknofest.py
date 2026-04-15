"""
crawl_teknofest.py
==================
Web crawler and ingestion pipeline for teknofest.org.

Changes from previous version
------------------------------
- Fixed broken `from langchain.text_splitter import …` import
- Uses shared EmbeddingService (centralized, no inline embedding logic)
- Uses shared text_cleaner + chunker pipeline
- Adds crawl_source (URL) and crawled_at timestamp metadata per chunk
- Deduplicates on re-crawl via content_hash
- Configurable crawl depth via MAX_DEPTH

Run
---
::

    python scripts/crawl_teknofest.py
"""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Set

import httpx
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.chunker import chunk_documents
from app.rag.embedding_service import get_embedding_service
from app.rag.text_cleaner import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("crawl_teknofest")

# ---------------------------------------------------------------------------
# Crawler configuration
# ---------------------------------------------------------------------------

START_URL = "https://teknofest.org/tr/"
MAX_DEPTH = 2
TIMEOUT = 15.0
SLEEP_BETWEEN_REQUESTS = 0.6


def is_same_domain(url: str) -> bool:
    return "teknofest.org" in url


def normalize_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------


def crawl() -> List[Document]:
    """BFS crawl of teknofest.org up to MAX_DEPTH. Returns LangChain Documents."""
    client = httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    visited: Set[str] = set()
    queue = deque([(START_URL, 0)])
    docs: List[Document] = []

    while queue:
        url, depth = queue.popleft()
        url = normalize_url(url)

        if url in visited or depth > MAX_DEPTH:
            continue
        visited.add(url)

        if not is_same_domain(url):
            continue

        try:
            logger.info("[depth=%d] GET %s", depth, url)
            resp = client.get(url)
            if resp.status_code != 200:
                logger.warning("  → status %d", resp.status_code)
                continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("  → request error: %s", exc)
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        page_title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else url
        )

        # Remove script / style / noscript noise
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.extract()

        raw_text = " ".join(soup.get_text(separator=" ").split())

        if raw_text.strip():
            docs.append(
                Document(
                    page_content=raw_text,
                    metadata={
                        "source": url,
                        "url": url,
                        "title": page_title,
                        "document_type": "web",
                    },
                )
            )

        # Enqueue links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            if href.startswith("/"):
                next_url = "https://teknofest.org" + href
            elif href.startswith("http"):
                next_url = href
            else:
                continue
            next_url = normalize_url(next_url)
            if is_same_domain(next_url) and next_url not in visited:
                queue.append((next_url, depth + 1))

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    client.close()
    logger.info("Crawl complete — %d pages collected.", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Deduplication helper (same as in ingest_local_docs.py)
# ---------------------------------------------------------------------------


def get_existing_hashes(chroma_path: Path, collection_name: str) -> Set[str]:
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
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------


def build_index(docs: List[Document]) -> None:
    settings = get_settings()
    settings.chroma_teknofest_site_path.mkdir(parents=True, exist_ok=True)

    embedding_svc = get_embedding_service(settings)
    lc_embeddings = embedding_svc.get_langchain_embeddings()

    existing_hashes = get_existing_hashes(
        settings.chroma_teknofest_site_path, "teknofest_site"
    )
    logger.info("Existing Chroma hashes: %d", len(existing_hashes))

    all_new_chunks = []
    total_skipped = 0

    for doc in docs:
        url = doc.metadata.get("url", "unknown")
        chunks = chunk_documents(
            docs=[doc],
            doc_type="web",
            source_type="teknofest",
            min_chunk_size=settings.chunk_min_size,
            target_chunk_size=settings.chunk_target_size,
            max_chunk_size=settings.chunk_max_size,
            chunk_overlap=settings.chunk_overlap,
            crawl_source=url,
        )

        new_chunks = []
        for chunk in chunks:
            h = chunk.metadata.get("content_hash", "")
            if h and h in existing_hashes:
                total_skipped += 1
                continue
            existing_hashes.add(h)
            new_chunks.append(chunk)

        all_new_chunks.extend(new_chunks)

    if not all_new_chunks:
        logger.info(
            "All %d chunks already indexed. Nothing to upsert.", total_skipped
        )
        return

    logger.info("Upserting %d new chunks into Chroma …", len(all_new_chunks))

    Chroma.from_documents(
        documents=all_new_chunks,
        embedding=lc_embeddings,
        persist_directory=str(settings.chroma_teknofest_site_path),
        collection_name="teknofest_site",
    )

    logger.info(
        "✓ Chroma 'teknofest_site' updated — %d new, %d skipped (dup).",
        len(all_new_chunks),
        total_skipped,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    docs = crawl()
    if not docs:
        logger.warning("No documents crawled.")
        return
    build_index(docs)


if __name__ == "__main__":
    main()
