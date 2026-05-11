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
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_hallucination_provider: str | None = Field(default=None)
    llm_hallucination_model: str | None = Field(default=None)
    llm_tavily_provider: str | None = Field(default=None)
    llm_tavily_model: str | None = Field(default=None)
    llm_reranker_provider: str | None = Field(default=None)
    llm_reranker_model: str | None = Field(default=None)

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
    rag_confidence_threshold: float = Field(default=0.40)

    # Chroma
    chroma_local_docs_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG" / "chroma_local_docs")
    chroma_teknofest_site_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "RAG" / "chroma_teknofest_site")

    # Chroma Collection İsimleri
    chroma_local_collection: str = Field(default="local_docs")
    chroma_site_collection: str = Field(default="teknofest_site")

    # ---- Retrieval ----
    # Candidates fetched from Chroma before reranking
    retrieval_top_k: int = Field(default=20)
    # Final chunks passed to the LLM after reranking / compression
    retrieval_final_k: int = Field(default=7)

    # ---- Reranker ----
    reranker_enabled: bool = Field(default=True)

    # ---- RAG Confidence Tuning ----
    rag_hard_floor_score: float = Field(default=0.45)
    rag_hard_floor_confidence: float = Field(default=0.56)
    rag_top_n_for_confidence: int = Field(default=3)

    # ---- Tavily Domain Filtering ----
    tavily_use_domain_filter: bool = Field(default=True)
    tavily_trusted_domains: list[str] = Field(default_factory=lambda: [
        "teknofest.org",
        "cdn.teknofest.org",
        "tubitak.gov.tr",
        "sanayi.gov.tr",
        "msb.gov.tr",
        "turkiyemaarif.gov.tr",
        "savunmasanayii.gov.tr",
        "aselsan.com.tr",
        "roketsan.com.tr",
        "tai.com.tr",
    ])

    # ---- Chunking ----
    chunk_min_size: int = Field(default=500)
    chunk_target_size: int = Field(default=2000)
    chunk_max_size: int = Field(default=2500)
    chunk_overlap: int = Field(default=400)

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

    # ---- Langfuse Observability (Module E) ----
    langfuse_enabled: bool = Field(default=False)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_public_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

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
        load_dotenv(dotenv_path, override=True)

    # OpenAI SDK reads OPENAI_BASE_URL directly from os.environ.
    # An empty string causes "UnsupportedProtocol" errors, so remove it.
    for key in ("OPENAI_BASE_URL",):
        if os.getenv(key) == "":
            os.environ.pop(key, None)

    return Settings(
        base_dir=base_dir,
        rag_root=rag_root,
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        llm_hallucination_provider=os.getenv("LLM_HALLUCINATION_PROVIDER"),
        llm_hallucination_model=os.getenv("LLM_HALLUCINATION_MODEL"),
        llm_tavily_provider=os.getenv("LLM_TAVILY_PROVIDER"),
        llm_tavily_model=os.getenv("LLM_TAVILY_MODEL"),
        llm_reranker_provider=os.getenv("LLM_RERANKER_PROVIDER"),
        llm_reranker_model=os.getenv("LLM_RERANKER_MODEL"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        kimi_api_key=os.getenv("KIMI_API_KEY"),
        kimi_base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        rag_confidence_threshold=float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.40")),
        chroma_local_docs_path=rag_root / "chroma_local_docs",
        chroma_teknofest_site_path=rag_root / "chroma_teknofest_site",
        # Chroma Collection İsimleri
        chroma_local_collection=os.getenv("CHROMA_LOCAL_COLLECTION", "local_docs"),
        chroma_site_collection=os.getenv("CHROMA_SITE_COLLECTION", "teknofest_site"),
        # Retrieval
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "20")),
        retrieval_final_k=int(os.getenv("RETRIEVAL_FINAL_K", "7")),
        # Reranker
        reranker_enabled=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        # RAG Confidence Tuning
        rag_hard_floor_score=float(os.getenv("RAG_HARD_FLOOR_SCORE", "0.45")),
        rag_hard_floor_confidence=float(os.getenv("RAG_HARD_FLOOR_CONFIDENCE", "0.56")),
        rag_top_n_for_confidence=int(os.getenv("RAG_TOP_N_FOR_CONFIDENCE", "3")),
        # Tavily Domain Filtering
        tavily_use_domain_filter=os.getenv("TAVILY_USE_DOMAIN_FILTER", "true").lower() == "true",
        tavily_trusted_domains=os.getenv("TAVILY_TRUSTED_DOMAINS").split(",") if os.getenv("TAVILY_TRUSTED_DOMAINS") else [
            "teknofest.org",
            "cdn.teknofest.org",
            "tubitak.gov.tr",
            "sanayi.gov.tr",
            "msb.gov.tr",
            "turkiyemaarif.gov.tr",
            "savunmasanayii.gov.tr",
            "aselsan.com.tr",
            "roketsan.com.tr",
            "tai.com.tr",
        ],
        # Chunking
        chunk_min_size=int(os.getenv("CHUNK_MIN_SIZE", "500")),
        chunk_target_size=int(os.getenv("CHUNK_TARGET_SIZE", "2000")),
        chunk_max_size=int(os.getenv("CHUNK_MAX_SIZE", "2500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "400")),
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
        # Langfuse (Module E)
        langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

