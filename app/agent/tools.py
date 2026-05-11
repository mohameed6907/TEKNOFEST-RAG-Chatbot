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
from langsmith import traceable


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
        "Mevcut hazır kategori örnekleri: Drone, Robotik, Yazılım, Yapay Zeka, İnsansız Hava Aracı. "
        "Bu kategorilerden birini yazarsan hemen tarihleri paylaşırım."
    )


@tool
def get_current_teknofest_news(query: str) -> str:
    """
    TEKNOFEST ile ilgili güncel haberleri, duyuruları, yarışma sonuçlarını ve kazananları web'den arar.
    Kullanıcı 'son haberler', 'yeni duyurular', 'bu yıl ne değişti', 'güncel',
    'kazananlar kim', 'birinci kim oldu', '2024 sonuçları', '2025 sonuçları' gibi 
    spesifik yıl veya sonuç soruları sorduğunda çağır.

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
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
        except ImportError:
            from langchain_tavily import TavilySearchResults  # type: ignore[import]
            
        TRUSTED_DOMAINS = ["teknofest.org", "t3vakfi.org", "trthaber.com", "aa.com.tr"]

        def search_trusted_sources(query: str, max_results: int = 7) -> list:
            tool = TavilySearchResults(max_results=max_results)
            
            # Try include_domains first
            try:
                results = tool.invoke({"query": query, "include_domains": TRUSTED_DOMAINS})
            except Exception:
                # Fallback: append site filter to query
                site_filter = " OR ".join([f"site:{d}" for d in TRUSTED_DOMAINS])
                results = tool.invoke(f"{query} ({site_filter})")
            
            # Sort by domain priority regardless
            def domain_rank(url: str) -> int:
                for i, domain in enumerate(TRUSTED_DOMAINS):
                    if domain in url:
                        return i
                return 99
            
            results = results.get("results", []) if isinstance(results, dict) else results
            results.sort(key=lambda r: domain_rank(r.get("url", "")))
            
            # Filter out results with rank 99 (untrusted domains) entirely
            trusted_results = [r for r in results if domain_rank(r.get("url", "")) < 99]
            
            # If trusted filter returns nothing, fall back to all results but still sorted
            return trusted_results if trusted_results else results

        enriched_query = f"TEKNOFEST {datetime.now().year} {query}"
        results_list = search_trusted_sources(enriched_query)

        if not results_list:
            return "Güncel haber bulunamadı."

        summaries: list[str] = []
        for r in results_list:
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
        "Şehir adını İstanbul, Ankara, İzmir veya Trabzon olarak yazarsan net lokasyon bilgisini hemen paylaşırım."
    )

@tool
@traceable(
    run_type="tool",
    name="fetch_teknofest_page", 
    metadata={"pipeline_stage": "live_fetch"}
)
def fetch_teknofest_competition_page(competition_slug: str) -> str:
    """
    Directly fetches and returns the clean text content of a TEKNOFEST competition page.
    Use this when the user asks about prizes, winners, deadlines, or rules for a specific competition.
    This is the PRIMARY source for prize amounts, application deadlines, and rankings.
    Always prefer this tool over VectorDB context for time-sensitive data.

    Args:
        competition_slug: the URL slug of the competition page,
                          e.g. 'insansiz-kara-araci-yarismasi'
    """
    import httpx
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        """Minimal HTML → plain-text extractor (stdlib only, no extra deps)."""
        def __init__(self):
            super().__init__()
            self._skip = False
            self._skip_tags = {"script", "style", "noscript", "head", "nav", "footer", "header"}
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in self._skip_tags:
                self._skip = True

        def handle_endtag(self, tag):
            if tag in self._skip_tags:
                self._skip = False
            # Add newlines after block elements to preserve structure
            if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "div", "br"}:
                self.parts.append("\n")

        def handle_data(self, data):
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self.parts.append(stripped + " ")

        def get_text(self) -> str:
            import re
            text = "".join(self.parts)
            # Collapse multiple blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

    url = f"https://www.teknofest.org/tr/yarismalar/{competition_slug}/"
    try:
        response = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TEKNOFESTBot/2.0)"},
        )
        if response.status_code != 200:
            return (
                f"[FETCH ERROR] {url} returned HTTP {response.status_code}. "
                "Sayfa mevcut olmayabilir veya slug yanlış olabilir."
            )

        parser = _TextExtractor()
        parser.feed(response.text)
        clean_text = parser.get_text()

        # Limit to 10 000 chars to stay within context window
        if len(clean_text) > 10_000:
            clean_text = clean_text[:10_000] + "\n\n[... metin kesildi ...]"

        if len(clean_text) < 100:
            # Possibly a JS-rendered SPA — return partial raw HTML as fallback
            return (
                f"URL: {url}\n"
                "[UYARI] Sayfa içeriği çok kısa — sunucu tarafında render edilmiş olabilir.\n"
                f"Ham HTML (ilk 5000 karakter):\n{response.text[:5_000]}"
            )

        return f"KAYNAK: {url} (canlı veri — {datetime.now().strftime('%Y-%m-%d')})\n\n{clean_text}"

    except Exception as e:
        return f"[FETCH ERROR] {url} için sayfa çekilemedi: {e}"
