from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import get_settings


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


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
    return loader.load()


def hash_path(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def main():
    settings = get_settings()
    raw_root = settings.rag_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    settings.chroma_local_docs_path.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    all_docs = []
    for file_path in iter_files(raw_root):
        docs = load_document(file_path)
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = str(file_path)
            d.metadata["doc_id"] = hash_path(file_path)
        all_docs.extend(docs)

    if not all_docs:
        print("No documents found in", raw_root)
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(all_docs)

    print(f"Loaded {len(all_docs)} docs, split into {len(split_docs)} chunks.")

    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(settings.chroma_local_docs_path),
        collection_name="local_docs",
    )

    print("Chroma index for local_docs updated at", settings.chroma_local_docs_path)


if __name__ == "__main__":
    main()

