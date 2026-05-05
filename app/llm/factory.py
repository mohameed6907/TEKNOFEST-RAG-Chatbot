from __future__ import annotations

from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import Settings


class LLMService(Protocol):
    def get_chat_model(self, *, temperature: float = 0.2, purpose: str = "main") -> BaseChatModel:
        ...


class UniversalLLMService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_chat_model(self, *, temperature: float = 0.2, purpose: str = "main") -> BaseChatModel:
        if purpose == "hallucination":
            provider = self.settings.llm_hallucination_provider or self.settings.llm_provider
            model = self.settings.llm_hallucination_model or self.settings.llm_model
        elif purpose == "tavily":
            provider = self.settings.llm_tavily_provider or self.settings.llm_provider
            model = self.settings.llm_tavily_model or self.settings.llm_model
        else:
            provider = self.settings.llm_provider
            model = self.settings.llm_model

        provider = provider.lower().strip()

        if provider == "groq":
            if not self.settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set")
            return ChatGroq(model=model, api_key=self.settings.groq_api_key, temperature=temperature)
        
        elif provider == "deepseek":
            if not self.settings.deepseek_api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set")
            return ChatOpenAI(model=model, api_key=self.settings.deepseek_api_key, base_url=self.settings.deepseek_base_url, temperature=temperature)
        
        elif provider == "kimi":
            if not self.settings.kimi_api_key:
                raise RuntimeError("KIMI_API_KEY is not set")
            return ChatOpenAI(model=model, api_key=self.settings.kimi_api_key, base_url=self.settings.kimi_base_url, temperature=temperature)
        
        elif provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            return ChatOpenAI(model=model, api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url, temperature=temperature)
        
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def get_llm_service(settings: Settings) -> LLMService:
    return UniversalLLMService(settings)

