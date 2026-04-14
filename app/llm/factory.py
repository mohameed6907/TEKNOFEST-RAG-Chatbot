from __future__ import annotations

from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import Settings


class LLMService(Protocol):
    def get_chat_model(self, *, temperature: float = 0.2) -> BaseChatModel:
        ...


class GroqService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_chat_model(self, *, temperature: float = 0.2) -> BaseChatModel:
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return ChatGroq(
            model=self.settings.llm_model,
            api_key=self.settings.groq_api_key,
            temperature=temperature,
        )


class OpenAICompatibleService:
    """
    DeepSeek/Kimi/OpenAI gibi OpenAI-compatible API'ler için ortak katman.
    """

    def __init__(self, *, api_key: str | None, model: str, base_url: str | None, key_name: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.key_name = key_name

    def get_chat_model(self, *, temperature: float = 0.2) -> BaseChatModel:
        if not self.api_key:
            raise RuntimeError(f"{self.key_name} is not set")
        kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": temperature,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)


def get_llm_service(settings: Settings) -> LLMService:
    provider = settings.llm_provider.lower().strip()
    if provider == "groq":
        return GroqService(settings)
    if provider == "deepseek":
        return OpenAICompatibleService(
            api_key=settings.deepseek_api_key,
            model=settings.llm_model,
            base_url=settings.deepseek_base_url,
            key_name="DEEPSEEK_API_KEY",
        )
    if provider == "kimi":
        return OpenAICompatibleService(
            api_key=settings.kimi_api_key,
            model=settings.llm_model,
            base_url=settings.kimi_base_url,
            key_name="KIMI_API_KEY",
        )
    if provider == "openai":
        return OpenAICompatibleService(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.openai_base_url,
            key_name="OPENAI_API_KEY",
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

