from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Base
    app_name: str = "TEKNOFEST RAG Chatbot"
    environment: str = Field(default="development")

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    rag_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG")

    # LLM Provider
    llm_provider: str = Field(default="groq")
    llm_model: str = Field(default="llama-3.1-70b-versatile")

    # Provider API keys / base urls
    groq_api_key: str | None = Field(default=None)
    deepseek_api_key: str | None = Field(default=None)
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1")
    kimi_api_key: str | None = Field(default=None)
    kimi_base_url: str = Field(default="https://api.moonshot.ai/v1")
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)

    # Embeddings (HuggingFace)
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Tavily
    tavily_api_key: str | None = Field(default=None)

    # Retrieval guard
    rag_confidence_threshold: float = Field(default=0.55)

    # Chroma
    chroma_local_docs_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG" / "chroma_local_docs")
    chroma_teknofest_site_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG" / "chroma_teknofest_site")

    class Config:
        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load settings from environment variables.

    Uses standard env vars (subset):
      - LLM_PROVIDER, LLM_MODEL
      - GROQ_API_KEY / DEEPSEEK_API_KEY / KIMI_API_KEY / OPENAI_API_KEY
      - EMBEDDING_MODEL_NAME
      - TAVILY_API_KEY
    """
    base_dir = Path(__file__).resolve().parent.parent
    rag_root = base_dir / "RAG"

    # Proje kökündeki .env dosyasını yükle
    dotenv_path = base_dir / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

    return Settings(
        base_dir=base_dir,
        rag_root=rag_root,
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        kimi_api_key=os.getenv("KIMI_API_KEY"),
        kimi_base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        rag_confidence_threshold=float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.55")),
        chroma_local_docs_path=rag_root / "chroma_local_docs",
        chroma_teknofest_site_path=rag_root / "chroma_teknofest_site",
    )

