from app.config import get_settings
from app.rag.graph import build_teknofest_graph, run_graph

import pytest
import asyncio


def test_intent_for_teknofest_question(monkeypatch):
    """
    Basit bir smoke test: TEKNOFEST içeren bir soruda graph çalışabilmeli.
    (Bu test gerçek LLM API anahtarı olmadan başarısız olabilir, bu nedenle
    CI ortamında isteğe bağlı tutulmalıdır.)
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()
    required_key = {
        "groq": settings.groq_api_key,
        "deepseek": settings.deepseek_api_key,
        "kimi": settings.kimi_api_key,
        "openai": settings.openai_api_key,
    }.get(provider)
    if (not required_key) or required_key.startswith("your-"):
        pytest.skip(f"Skipping test: provider key missing for {provider}")
    graph = build_teknofest_graph(settings)
    # Bu sadece çalışıp çalışmadığını kontrol eder; cevap içeriğini doğrulamaz.
    result = asyncio.run(run_graph(graph, "TEKNOFEST nerede yapılıyor?"))
    assert "answer" in result

