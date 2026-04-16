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

    # Embeddings (OpenAI only)
    embedding_provider: str = Field(default="openai")
    embedding_model_name: str = Field(default="text-embedding-3-small")

    # Tavily
    tavily_api_key: str | None = Field(default=None)

    # Retrieval guard
    rag_confidence_threshold: float = Field(default=0.55)

    # Chroma
    chroma_local_docs_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG" / "chroma_local_docs")
    chroma_teknofest_site_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG" / "chroma_teknofest_site")

    # ---- Retrieval ----
    # Candidates fetched from Chroma before reranking
    retrieval_top_k: int = Field(default=10)
    # Final chunks passed to the LLM after reranking / compression
    retrieval_final_k: int = Field(default=5)

    # ---- Reranker ----
    reranker_enabled: bool = Field(default=True)

    # ---- Chunking ----
    chunk_min_size: int = Field(default=400)
    chunk_target_size: int = Field(default=800)
    chunk_max_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=150)

    # ---- Evaluation / Logging ----
    eval_log_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG" / "eval_log.jsonl"
    )
    eval_dataset_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
    )

    # ---- LangSmith Observability ----
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str = Field(default="teknofest-rag")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")

    class Config:
        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load settings from environment variables.

    Uses standard env vars (subset):
      - LLM_PROVIDER, LLM_MODEL
      - GROQ_API_KEY / DEEPSEEK_API_KEY / KIMI_API_KEY / OPENAI_API_KEY
      - EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME
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
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        rag_confidence_threshold=float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.55")),
        chroma_local_docs_path=rag_root / "chroma_local_docs",
        chroma_teknofest_site_path=rag_root / "chroma_teknofest_site",
        # Retrieval
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
        retrieval_final_k=int(os.getenv("RETRIEVAL_FINAL_K", "5")),
        # Reranker
        reranker_enabled=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        # Chunking
        chunk_min_size=int(os.getenv("CHUNK_MIN_SIZE", "400")),
        chunk_target_size=int(os.getenv("CHUNK_TARGET_SIZE", "800")),
        chunk_max_size=int(os.getenv("CHUNK_MAX_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        # Evaluation
        eval_log_path=Path(os.getenv("EVAL_LOG_PATH", str(rag_root / "eval_log.jsonl"))),
        eval_dataset_path=Path(
            os.getenv("EVAL_DATASET_PATH", str(base_dir / "data" / "eval_dataset.json"))
        ),
        # LangSmith
        langsmith_tracing=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        langsmith_api_key=os.getenv("LANGCHAIN_API_KEY"),
        langsmith_project=os.getenv("LANGCHAIN_PROJECT", "teknofest-rag"),
        langsmith_endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )

