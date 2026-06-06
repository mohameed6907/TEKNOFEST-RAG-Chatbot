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
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

from app.agent.tools import (
    get_current_teknofest_news,
    get_festival_location_info,
    get_teknofest_deadline_info,
    fetch_teknofest_competition_page,
)
from app.llm import get_llm_service

logger = logging.getLogger(__name__)

TOOLS = [
    get_teknofest_deadline_info,
    get_current_teknofest_news,
    get_festival_location_info,
    fetch_teknofest_competition_page,
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

CEVAP KURALLARI — BU KURALLARI KESİNLİKLE UYGULA:
1. Sağlanan bağlamdan TÜM ilgili bilgileri kullan — kısa tutma, kapsamlı ol
2. Bilgiyi mantıklı bölümlere ayır: başlık, madde listesi, tablo formatı kullan
3. Tarih, yer, ödül, kategori gibi somut veriler varsa mutlaka belirt
4. Bağlamda olmayan bir bilgiyi uydurma — ama bağlamda olan her şeyi kullan
5. Cevabın sonunda kullanıcıya faydalı olacak bir sonraki adımı öner
6. Her zaman Türkçe cevapla
7. "Daha fazla bilgi için belirtin" ile bitirme — zaten elindeki bilgiyi ver

ÖNEMLİ KURAL: Kullanıcı spesifik bir yarışmanın ödüllerini, kazananlarını, tarihlerini, yerini veya kurallarını soruyorsa, bağlamda (context) eski veya yeni bir cevap görsen BİLE, ASLA doğrudan cevap verme. MUTLAKA ÖNCE `fetch_teknofest_competition_page` aracını kullanarak o yarışmanın güncel web sayfasını çek ve cevabı oradan ver. Aracın argümanı yarışmanın slug'ı olmalıdır (örnek: 'insansiz-kara-araci-yarismasi').

YASAK İFADELER — BU İFADELERİ ASLA KULLANMA:
- "Daha fazla bilgi için [X sitesini] ziyaret edebilirsin/edebilirsiniz"
- "Resmi web sitesine bakmanı öneririm"
- "[URL] adresinden ulaşabilirsin"
- "teknofest.org adresini kontrol edebilirsin"
- "resmi kaynakları incelemenizi tavsiye ederim"
- Cevap sonuna dış kaynak linki veya site yönlendirmesi ekleme

KURAL: Sen zaten o kaynaklara erişiyorsun. Kullanıcıyı dışarı yönlendirmek yerine, bilgi eksikse fallback zincirini çalıştır (Site Crawl → Tavily → Grounded LLM). Cevabın içinde hiçbir zaman dış site yönlendirmesi olmasın. Eğer bilgi bulunamıyorsa "Bu konuda elimde yeterli bilgi bulunmuyor" de, dış siteye yönlendirme.

Bağlam boş veya tamamen yetersizse:
→ Açıkça bilgi bulamadığını belirt. Kullanıcıyı dış siteye yönlendirmek yerine, araçları (tools) kullanarak bilgiyi bulmaya çalış.

Bağlam kısmiysa:
→ YALNIZCA desteklenen kısımları cevapla.
→ Bağlamdan cevaplayamadığın kısımları açıkça belirt."""


@traceable(run_type="llm", name="generate_response")
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
        f"SAĞLANAN BAĞLAM:\n{context_str}\n\n"
        f"Kullanıcı sorusu: {question}\n\n"
        "Eğer bilgi yetersizse veya spesifik bir yarışma detayı isteniyorsa uygun araçları (tools) kullan.\n"
        "HATIRLATMA: Cevabında kullanıcıyı dış siteye yönlendirme, site linki verme veya 'ziyaret edin' gibi ifadeler kullanma."
    )
    messages.append(HumanMessage(content=user_prompt))

    lower_q = question.lower()
    needs_competition_tool = any(k in lower_q for k in [
        "ödül", "odul", "kazanan", "birinci", "ikinci", "üçüncü", "sonuç", "şampiyon",
        "kim oldu", "kim kazandı", "kimler",
        "nerede", "hangi şehir", "hangi sehir", "mekan", "konum", "venue",
        "ne zaman", "tarih", "deadline", "son başvuru", "son basvuru", "başvuru tarihi", "basvuru tarihi",
        "katılım koşulları", "katilim kosullari", "katılım şartları", "katilim sartlari",
        "kuralları", "kurallari", "nasıl katılırım", "nasil katilirim",
    ])

    for iteration in range(MAX_ITERATIONS):
        try:
            # Sadece ilk iterasyonda, spesifik yarışma sorusu varsa aracı zorla
            if iteration == 0 and needs_competition_tool:
                llm_to_use = llm.bind_tools(TOOLS, tool_choice="fetch_teknofest_competition_page")
                logger.info("Zorunlu araç çağrısı aktif: fetch_teknofest_competition_page")
            else:
                llm_to_use = llm_with_tools

            response: AIMessage = await llm_to_use.ainvoke(messages, config={"tags": ["final_synthesizer"]})
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent LLM çağrısı başarısız (iter %d): %s", iteration, exc)
            return "Şu an cevap üretiminde teknik bir aksaklık oluştu. Sorunu giderirken aynı soruyu biraz daha kısa yazarak tekrar deneyebilirsin."

        # Tool çağrısı yok → final cevap
        if not getattr(response, "tool_calls", None):
            return _strip_redirect_phrases(response.content or "")

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
        final: AIMessage = await llm_with_tools.ainvoke(messages, config={"tags": ["final_synthesizer"]})
        return _strip_redirect_phrases(final.content or "Bu soruyu mevcut bağlamla netleştiremedim. İstersen kategori veya yıl bilgisini ekleyerek tekrar sor; daha nokta atışı cevaplayayım.")
    except Exception:  # noqa: BLE001
        return "Bu soruyu mevcut bağlamla netleştiremedim. İstersen kategori veya yıl bilgisini ekleyerek tekrar sor; daha nokta atışı cevaplayayım."


# ---------------------------------------------------------------------------
# Post-processing: Strip redirect phrases from LLM output
# ---------------------------------------------------------------------------

_REDIRECT_PATTERNS = [
    # "Daha fazla bilgi için ... ziyaret edebilirsin/edebilirsiniz"
    r"[Dd]aha fazla bilgi için[^.!\n]*(?:ziyaret|bak|kontrol|incele|ulaş)[^.!\n]*[.!]?",
    # "Resmi web sitesini/sayfasını ziyaret edebilirsin"
    r"[Rr]esmi[^.!\n]*(?:site|sayfa|kaynak)[^.!\n]*(?:ziyaret|bak|kontrol|incele|ulaş)[^.!\n]*[.!]?",
    # "teknofest.org adresinden/adresini ..."
    r"[Tt]eknofest\.org[^.!\n]*(?:adres|kontrol|ziyaret|bak|incele|ulaş)[^.!\n]*[.!]?",
    # "... adresinden ulaşabilirsin"
    r"[^.!\n]*adresinden[^.!\n]*ulaşabilir[^.!\n]*[.!]?",
    # "... sitesini kontrol etmenizi öneririm"
    r"[^.!\n]*sitesini[^.!\n]*(?:kontrol|ziyaret|incele)[^.!\n]*(?:öneririm|tavsiye)[^.!\n]*[.!]?",
    # "... sayfasını ziyaret ..."
    r"[^.!\n]*sayfasını[^.!\n]*ziyaret[^.!\n]*[.!]?",
    # Standalone URL references like "teknofest.org/tr/yarismalar"
    r"(?:^|\n)[^.!\n]*teknofest\.org/[^\s]*[^.!\n]*[.!]?",
]


def _strip_redirect_phrases(text: str) -> str:
    """Remove any redirect-to-external-site phrases from the answer."""
    if not text:
        return text
    cleaned = text
    for pattern in _REDIRECT_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    # Clean up extra whitespace / newlines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text  # never return empty
