"""
tools.py
========
LangChain @tool tanımları — TEKNOFEST agentic pipeline (Module F).

Araçlar:
  - get_teknofest_deadline_info  : Yarışma başvuru tarihleri
  - get_current_teknofest_news   : Güncel TEKNOFEST haberleri (Tavily)
  - get_festival_location_info   : Festival mekan ve ulaşım bilgisi

Kritik kural: Bu araçlar yalnızca purpose="main" LLM'e bind_tools() ile
bağlanır. purpose="hallucination" veya purpose="tavily" LLM'lere asla
bind_tools() çağrılmaz.
"""
from __future__ import annotations

from langchain_core.tools import tool
from datetime import datetime


@tool
def get_teknofest_deadline_info(category: str) -> str:
    """
    Belirli bir TEKNOFEST yarışma kategorisinin başvuru son tarihini ve
    önemli tarihlerini döndürür. Kullanıcı 'ne zaman başvurulur',
    'son başvuru tarihi', 'tarih', 'deadline' gibi sorular sorduğunda çağır.

    Args:
        category: Yarışma kategorisi adı (örn: 'Drone', 'Yazılım', 'Robotik')

    Returns:
        Kategori tarihleri veya bulunamadı mesajı
    """
    # Dinamik yıl — Sprint 6
    year = datetime.now().year
    DEADLINES: dict[str, str] = {
        "drone": f"Başvuru: 15 Şubat {year} | Proje teslimi: 1 Nisan {year} | Final: Ağustos {year}",
        "insansiz hava araci": f"Başvuru: 15 Şubat {year} | Proje teslimi: 1 Nisan {year} | Final: Ağustos {year}",
        "yazilim": f"Başvuru: 20 Şubat {year} | Demo: 15 Nisan {year} | Final: Ağustos {year}",
        "yazılım": f"Başvuru: 20 Şubat {year} | Demo: 15 Nisan {year} | Final: Ağustos {year}",
        "robotik": f"Başvuru: 10 Şubat {year} | Ön eleme: 20 Mart {year} | Final: Ağustos {year}",
        "robot": f"Başvuru: 10 Şubat {year} | Ön eleme: 20 Mart {year} | Final: Ağustos {year}",
        "yapay zeka": f"Başvuru: 25 Şubat {year} | Sunum: 10 Nisan {year} | Final: Ağustos {year}",
        "ai": f"Başvuru: 25 Şubat {year} | Sunum: 10 Nisan {year} | Final: Ağustos {year}",
    }
    normalized = (
        category.lower()
        .strip()
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("ı", "i")
    )
    # Normalize anahtar listesinde de ara
    for key, value in DEADLINES.items():
        key_norm = (
            key.replace("ş", "s")
            .replace("ğ", "g")
            .replace("ü", "u")
            .replace("ö", "o")
            .replace("ç", "c")
            .replace("ı", "i")
        )
        if key_norm == normalized or key == category.lower().strip():
            return value

    return (
        f"'{category}' kategorisi için tarih bilgisi bulunamadı. "
        "Lütfen teknofest.org/tr/kategoriler/ adresini kontrol edin."
    )


@tool
def get_current_teknofest_news(query: str) -> str:
    """
    TEKNOFEST ile ilgili güncel haber ve duyuruları web'den arar.
    Kullanıcı 'son haberler', 'yeni duyurular', 'bu yıl ne değişti',
    'güncel' gibi sorular sorduğunda çağır.

    Args:
        query: Arama sorgusu (TEKNOFEST bağlamında otomatik zenginleştirilir)

    Returns:
        Güncel haber özeti
    """
    try:
        from langchain_tavily import TavilySearchResults  # type: ignore[import]
    except ImportError:
        return "Haber arama servisi şu an kullanılamıyor (langchain-tavily kurulu değil)."

    try:
        tavily = TavilySearchResults(max_results=3)
        enriched_query = f"TEKNOFEST {datetime.now().year} {query}"
        results = tavily.invoke(enriched_query)

        if not results:
            return "Güncel haber bulunamadı."

        summaries: list[str] = []
        for r in results:
            title = r.get("title", "Başlıksız")
            content = r.get("content", "")[:200]
            summaries.append(f"- {title}: {content}...")
        return "\n".join(summaries)
    except Exception as exc:  # noqa: BLE001
        return f"Haber araması başarısız: {exc}"


@tool
def get_festival_location_info(city: str = "İstanbul") -> str:
    """
    TEKNOFEST festival alanı bilgilerini döndürür: şehir, mekan, ulaşım.
    Kullanıcı 'nerede yapılıyor', 'festival alanı', 'nasıl gidilir',
    'mekan', 'adres' gibi sorular sorduğunda çağır.

    Args:
        city: Festival şehri (varsayılan: İstanbul)

    Returns:
        Mekan ve ulaşım bilgisi
    """
    VENUES: dict[str, str] = {
        "istanbul": (
            "TEKNOFEST İstanbul — Atatürk Havalimanı Fuar Alanı. "
            "Metro: M1 Havalimanı durağı. Ücretsiz giriş."
        ),
        "ankara": (
            "TEKNOFEST Ankara — MKE Fabrikaları Alanı. "
            "Ankara Merkez'den servis mevcut."
        ),
        "izmir": (
            "TEKNOFEST İzmir — Fuar İzmir. "
            "İzban: Fuar İzmir durağı."
        ),
        "trabzon": (
            "TEKNOFEST Trabzon — Trabzon Havalimanı çevresi. "
            "Şehir merkezinden servis mevcut."
        ),
    }
    normalized = (
        city.lower()
        .strip()
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("ı", "i")
    )
    for key, value in VENUES.items():
        if key == normalized or key in normalized:
            return value

    return (
        f"{city} için mekan bilgisi bulunamadı. "
        "Lütfen teknofest.org adresini kontrol edin."
    )
