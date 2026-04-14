from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Set

import httpx
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import get_settings


START_URL = "https://teknofest.org/tr/"
MAX_DEPTH = 2
TIMEOUT = 15.0
SLEEP_BETWEEN_REQUESTS = 0.6


def is_same_domain(url: str) -> bool:
    return "teknofest.org" in url


def normalize_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


def crawl() -> List[Document]:
    settings = get_settings()
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
            print(f"[{depth}] GET {url}")
            resp = client.get(url)
            if resp.status_code != 200:
                print("  -> status", resp.status_code)
                continue
        except Exception as exc:
            print("  -> error", exc)
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # basic text extraction
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        text = " ".join(soup.get_text(separator=" ").split())

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "url": url,
                        "title": title,
                        "source": url,
                    },
                )
            )

        # queue links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("mailto:") or href.startswith("tel:"):
                continue
            if href.startswith("/"):
                next_url = "https://teknofest.org" + href
            else:
                next_url = href
            next_url = normalize_url(next_url)
            if is_same_domain(next_url) and next_url not in visited:
                queue.append((next_url, depth + 1))

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    client.close()
    return docs


def build_index(docs: List[Document]) -> None:
    settings = get_settings()
    settings.chroma_teknofest_site_path.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
    )
    split_docs = splitter.split_documents(docs)

    print(f"Split {len(docs)} docs into {len(split_docs)} chunks.")

    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(settings.chroma_teknofest_site_path),
        collection_name="teknofest_site",
    )

    print("Chroma index for teknofest_site updated at", settings.chroma_teknofest_site_path)


def main():
    docs = crawl()
    if not docs:
        print("No documents crawled.")
        return
    build_index(docs)


if __name__ == "__main__":
    main()

