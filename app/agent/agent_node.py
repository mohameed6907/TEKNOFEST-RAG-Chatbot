"""
agent_node.py
=============
Tool-calling agent loop for the TEKNOFEST RAG pipeline (Module F).

run_agent_node():
  - purpose="main" LLM'e bind_tools() uygular
  - Max 3 iterasyon (sonsuz döngü koruması)
  - Tool çağrısı yoksa direkt LLM cevabı döner
  - Hata durumunda güvenli fallback mesajı döner

Kritik: bind_tools() YALNIZCA bu modülde ve YALNIZCA purpose="main"
LLM'e uygulanır. purpose="hallucination" veya purpose="tavily"
LLM'lere asla bind_tools() çağrılmaz.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.tools import (
    get_current_teknofest_news,
    get_festival_location_info,
    get_teknofest_deadline_info,
)
from app.llm import get_llm_service

logger = logging.getLogger(__name__)

TOOLS = [
    get_teknofest_deadline_info,
    get_current_teknofest_news,
    get_festival_location_info,
]
TOOL_MAP = {t.name: t for t in TOOLS}

MAX_ITERATIONS = 3


def _build_agent_system_prompt() -> str:
    """Runtime'da güncel tarihle system prompt oluşturur."""
    from datetime import datetime
    from app.rag.graph import PERSONALITY_PREFIX
    current_date = datetime.now().strftime("%d %B %Y, %A")
    return f"""{PERSONALITY_PREFIX}

Bugünün tarihi: {current_date}

KATIŞTI RAG KURALLAR - BU KURALLARI KESINLIKLE TAŞIMALIN:

1. YALNIZCA sağlanan bağlamı kullan. Genel bilgiye ASLA güvenme.
2. ASLA harici bilgi veya web bilgisi kullanma.
3. ASLA bilgi uydurma, tahmin etme veya çıkarım yapma.
4. ASLA boşlukları doldurma veya bağlamda olmayan bilgiyle cevapla.
5. Her zaman Türkçe cevapla.
6. Her zaman bağlamdan kaynaklarını belirt.

Bağlam boş veya yetersizse:
→ MUTLAKA şu şekilde cevapla: "Sağlanan belgelerden ilgili bilgi bulamadım."

Bağlam kısmiysa:
→ YALNIZCA desteklenen kısımları cevapla.
→ Bağlamdan cevaplayamadığın kısımları açıkça belirt."""


async def run_agent_node(
    question: str,
    context_str: str,
    settings,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Tool-calling agent loop.

    Args:
        question:     Kullanıcının sorusu (rephrased_question tercih edilir)
        context_str:  RAG pipeline'ından gelen bağlam metni
        settings:     app.config.Settings nesnesi
        chat_history: Önceki mesajlar (opsiyonel, bağlam için)

    Returns:
        Nihai cevap string'i
    """
    llm = get_llm_service(settings).get_chat_model(temperature=0.1, purpose="main")

    # bind_tools() YALNIZCA purpose="main" LLM'e
    try:
        llm_with_tools = llm.bind_tools(TOOLS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bind_tools() başarısız, tool'suz devam ediliyor: %s", exc)
        llm_with_tools = llm

    # Mesaj geçmişini oluştur
    messages: list = [SystemMessage(content=_build_agent_system_prompt())]

    # Önceki konuşma bağlamını ekle (varsa)
    for msg in (chat_history or []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    # Kullanıcı sorusu + RAG bağlamı
    user_prompt = (
        f"SAĞLANAN BAĞLAM (YALNIZCA bunu kullan):\n{context_str}\n\n"
        f"Kullanıcı sorusu: {question}\n\n"
        "KATIŞTI: Cevabı YALNIZCA yukarıdaki bağlamdan ver. "
        "Genel bilgi KULLANMA. "
        "Bağlam boş veya yetersizse şu cümleyi yanıt ver: 'Sağlanan belgelerden ilgili bilgi bulamadım.'"
    )
    messages.append(HumanMessage(content=user_prompt))

    for iteration in range(MAX_ITERATIONS):
        try:
            response: AIMessage = await llm_with_tools.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent LLM çağrısı başarısız (iter %d): %s", iteration, exc)
            return "Şu an cevap üretemiyorum. Lütfen teknofest.org adresini ziyaret edin."

        # Tool çağrısı yok → final cevap
        if not getattr(response, "tool_calls", None):
            return response.content or ""

        # Tool'ları çalıştır
        messages.append(response)
        logger.debug("Agent iter %d — %d tool çağrısı", iteration, len(response.tool_calls))

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name not in TOOL_MAP:
                tool_result = f"Hata: '{tool_name}' aracı bulunamadı."
                logger.warning("Bilinmeyen tool: %s", tool_name)
            else:
                try:
                    tool_result = TOOL_MAP[tool_name].invoke(tool_args)
                    logger.debug("Tool '%s' sonucu: %s", tool_name, str(tool_result)[:100])
                except Exception as exc:  # noqa: BLE001
                    tool_result = f"Araç hatası ({tool_name}): {exc}"
                    logger.warning("Tool '%s' hata: %s", tool_name, exc)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

    # Max iterasyona ulaşıldı — son LLM cevabını almayı dene
    logger.warning("Agent max_iterations=%d'e ulaştı", MAX_ITERATIONS)
    try:
        final: AIMessage = await llm_with_tools.ainvoke(messages)
        return final.content or "Yeterli bilgiye ulaşılamadı. Lütfen teknofest.org adresini ziyaret edin."
    except Exception:  # noqa: BLE001
        return "Yeterli bilgiye ulaşılamadı. Lütfen teknofest.org adresini ziyaret edin."
