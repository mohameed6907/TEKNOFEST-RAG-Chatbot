"""
graph.py
========
LangGraph workflow for the TEKNOFEST RAG chatbot.

Pipeline (TEKNOFEST intent path)
---------------------------------
intent → local_rag → [teknofest_web] → [tavily_web]
       → reranker (optional) → context_builder → answer_synthesizer
       → hallucination_guard → END

Each node is a pure function / async function accepting and returning
GraphState.  Settings are injected via closure at graph construction time.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.llm import get_llm_service
from app.rag.context_builder import build_context
from app.rag.evaluator import Timer, log_retrieval_event, tag_failures
from app.rag.hallucination_guard import hallucination_check
from app.rag.memory import build_rephrase_chain, format_chat_history
from app.rag.prompts import INTENT_CLASSIFICATION_PROMPT, SYSTEM_PROMPT_BASE
from app.rag.reranker import rerank_chunks
from app.rag.text_cleaner import normalize_turkish_query
from app.rag.retrievers import (
    RetrievedChunk,
    build_local_docs_retriever,
    build_teknofest_site_retriever,
    build_tavily_tool,
    estimate_chunk_confidence,
    retrieve_from_tavily,
    retrieve_from_vectorstore,
)
from app.tracing import get_run_metadata, is_tracing_enabled
from langsmith import traceable

logger = logging.getLogger(__name__)

RouteLiteral = Literal["direct", "local", "site", "tavily", "llm_knowledge", "kisisel", "live_fetch"]

PERSONALITY_PREFIX = """Sen Türkiye'nin en büyük teknoloji festivali TEKNOFEST'in yardımcı chatbot'usun.
Kullanıcılarla samimi, sıcak ve destekleyici bir dilde konuşursun.
Sanki yarışmacıların en büyük destekçisiymiş gibi hissettirirsin.
Cevaplarını her zaman Türkçe verirsin.
Özellikle kategori veya takım sorularında yarışmacıların heyecanını paylaş."""

# ---------------------------------------------------------------------------
# Freshness gate — keywords and stale-source markers
# ---------------------------------------------------------------------------

# Query keywords that signal time-sensitive data (prizes, deadlines, rankings)
FRESHNESS_SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "ödül", "odul", "para ödülü", "para odulu",
    "birinci", "ikinci", "üçüncü", "ucuncu",
    "kazanan", "kazananlar", "şampiyon", "sampiyon",
    "derece", "dereceye",
    "başvuru tarihi", "basvuru tarihi", "son başvuru", "son basvuru",
    "deadline", "tarih",
    "katılımcı sayısı", "katilimci sayisi", "kota",
    "winner", "prize", "ranking",
)

# Source name substrings that indicate a potentially stale local document
STALE_SOURCE_MARKERS: tuple[str, ...] = (
    "2021", "2022", "2023", "2024",
    "Ansiklopedi", "ansiklopedi",
    "Kapsamli_Rehber", "Kapsamli Rehber",
    "TEKNOFEST_Kapsamli",
    "Bilgi_Veri", "Bilgi Veri",
    "genel.docx",
)

TAVILY_TRIGGER_KEYWORDS = (
    "bu yıl",
    "bu sene",
    "2023",
    "2024",
    "2025",
    "2026",
    "2027",
    "geçen yıl",
    "gecen yil",
    "güncel",
    "yeni",
    "current",
    "current year",
    "this year",
    "ödül",
    "odul",
    "ödül miktarı",
    "odul miktari",
    "ödül farkı",
    "odul farki",
    "para ödülü",
    "ucret",
    "ücret",
    "fiyat",
    "başvuru ücreti",
    "basvuru ucreti",
    "kayıt ücreti",
    "kayit ucreti",
    "ücret farkı",
    "ucret farki",
    "son başvuru",
    "son basvuru",
    "başvuru tarihi",
    "basvuru tarihi",
    "deadline",
    "tarih",
    "ne kadar",
    "maliyet",
    "kazanan",
    "kazananlar",
    "birinci",
    "ikinci",
    "üçüncü",
    "ucuncu",
    "derece",
    "şampiyon",
    "sampiyon",
    "sonuç",
    "sonuc",
    "kim oldu",
    "kim kazandı",
    "kim kazandi",
    "birincileri"
)

CATEGORY_ALIASES = (
    ("sağlıkta yapay zeka", "Sağlıkta Yapay Zeka"),
    ("saglikta yapay zeka", "Sağlıkta Yapay Zeka"),
    ("insansız hava aracı", "İnsansız Hava Aracı"),
    ("insansiz hava araci", "İnsansız Hava Aracı"),
    ("iha", "İnsansız Hava Aracı"),
    ("insansız kara aracı", "İnsansız Kara Aracı"),
    ("insansiz kara araci", "İnsansız Kara Aracı"),
    ("insansiz kara", "İnsansız Kara Aracı"),
    ("robotik", "Robotik"),
    ("robolig", "Robolig"),
    ("yazılım", "Yazılım"),
    ("yazilim", "Yazılım"),
    ("drone", "Drone"),
)

USER_OWNERSHIP_HINTS = (
    "ben",
    "kategorim",
    "alanim",
    "alanım",
    "yaris",
    "yarış",
    "hazirlaniyorum",
    "hazırlanıyorum",
    "sectim",
    "seçtim",
)

DISALLOWED_REDIRECT_PHRASES = (
    "resmi kaynak",
    "resmi site",
    "web sitesini kontrol",
    "adresini kontrol",
    "teknofest.org",
    "kontrol etmeni oneririm",
    "kontrol etmenizi oneririm",
    "ziyaret edin",
)



# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, str]]
    rephrased_question: str          # D.2 — rephrase node çıktısı
    intent: Literal["TEKNOFEST", "DIGER", "KISISEL"]
    # All candidates fetched from vector stores
    retrieved_chunks: List[RetrievedChunk]
    # Final context after reranking + compression
    context_chunks: List[RetrievedChunk]
    context_str: str
    answer: str
    route_taken: RouteLiteral
    meta: Dict[str, Any]
    # Timing
    _timer_start_ms: float


# ---------------------------------------------------------------------------
# LLM builder helper
# ---------------------------------------------------------------------------


def _build_llm(settings: Settings, temperature: float = 0.2, purpose: str = "main"):
    return get_llm_service(settings).get_chat_model(temperature=temperature, purpose=purpose)


def _needs_tavily(question: str) -> bool:
    normalized = normalize_turkish_query(question or "").lower()
    return any(keyword in normalized for keyword in TAVILY_TRIGGER_KEYWORDS)


def _find_categories_in_text(text: str) -> List[str]:
    normalized_text = normalize_turkish_query(text or "").lower()
    matches: List[tuple[int, int, str]] = []
    for alias, category_name in CATEGORY_ALIASES:
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        for m in re.finditer(pattern, normalized_text):
            matches.append((m.start(), -len(alias), category_name))

    matches.sort(key=lambda item: (item[0], item[1]))

    ordered_categories: List[str] = []
    seen = set()
    for _, _, category_name in matches:
        if category_name in seen:
            continue
        seen.add(category_name)
        ordered_categories.append(category_name)
    return ordered_categories


def _looks_like_user_category_statement(text: str) -> bool:
    normalized = normalize_turkish_query(text or "").lower()
    return any(hint in normalized for hint in USER_OWNERSHIP_HINTS)


def _contains_disallowed_redirect(answer: str) -> bool:
    normalized = normalize_turkish_query(answer or "").lower()
    return any(phrase in normalized for phrase in DISALLOWED_REDIRECT_PHRASES)


def _extract_category_context(chat_history: List[Dict[str, str]], question: str) -> str | None:
    history = chat_history or []

    # 1) Most reliable source: user's own explicit category statements, newest first.
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "") or ""
        if not _looks_like_user_category_statement(content):
            continue
        categories = _find_categories_in_text(content)
        if categories:
            return categories[0]

    # 2) Fallback: most recent user message containing any known category.
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        categories = _find_categories_in_text(msg.get("content", "") or "")
        if categories:
            return categories[0]

    # 3) Last resort: current question if user explicitly frames it as own category.
    if _looks_like_user_category_statement(question):
        categories = _find_categories_in_text(question)
        if categories:
            return categories[0]

    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def node_intent_classification(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.0, purpose="tavily")
    from datetime import datetime
    current_date = datetime.now().strftime("%d %B %Y, %A")
    query_to_classify = state.get("rephrased_question") or state["question"]
    prompt = f"Bugünün tarihi: {current_date}\n\n" + INTENT_CLASSIFICATION_PROMPT.format(question=query_to_classify)
    res = await llm.ainvoke([{"role": "user", "content": prompt}])
    label_raw = (res.content or "").strip().upper()
    # Normalize Turkish characters for robust matching
    label_normalized = (
        label_raw
        .replace("İ", "I").replace("Ğ", "G").replace("Ş", "S")
        .replace("Ü", "U").replace("Ö", "O").replace("Ç", "C")
    )
    logger.debug("Intent raw label: %r → normalized: %r", label_raw, label_normalized)
    if "KISISEL" in label_normalized or "KIŞISEL" in label_raw:
        intent = "KISISEL"
    elif "DIGER" in label_normalized or "DİĞER" in label_raw:
        intent = "DIGER"
    elif "TEKNOFEST" in label_normalized:
        intent = "TEKNOFEST"
    else:
        # Fallback: if LLM returned something unexpected, default to TEKNOFEST
        intent = "TEKNOFEST"
        logger.warning("Intent classification returned unexpected label: %r, defaulting to TEKNOFEST", label_raw)
    state["intent"] = intent
    state.setdefault("meta", {})["intent"] = intent
    return state


def _decide_next_after_intent(state: GraphState) -> str:
    if state.get("intent") == "TEKNOFEST":
        return "local_rag"
    elif state.get("intent") == "KISISEL":
        return "kisisel_llm"
    else:
        return "direct_llm"


# ---- Rephrase (D.3) ----

async def node_rephrase(state: GraphState, settings: Settings) -> GraphState:
    """
    D.3 — Chat history varsa soruyu bağımsız hale getirir.
    İlk mesajsa (history yok) orijinal soruyu doğrudan kullanır.
    """
    history = state.get("chat_history") or []
    
    # If no history, use the original question
    if not history or len(history) == 0:
        state["rephrased_question"] = state["question"]
        state.setdefault("meta", {})["rephrase_used"] = False
        return state

    # History exists — rephrase to make the question standalone
    llm = _build_llm(settings, temperature=0.0, purpose="rephrase")
    rephrase_chain = build_rephrase_chain(llm)
    history_str = format_chat_history(history)

    try:
        rephrased = await rephrase_chain.ainvoke({
            "chat_history": history_str,
            "question": state["question"],
        })
        rephrased = rephrased.strip()
        
        # Fallback: if LLM returned something too short or weird, use original
        if len(rephrased) < 3:
            state["rephrased_question"] = state["question"]
            state.setdefault("meta", {})["rephrase_used"] = False
            state.setdefault("meta", {})["rephrase_fallback"] = "output_too_short"
        else:
            state["rephrased_question"] = rephrased
            state.setdefault("meta", {})["rephrase_used"] = True
            state.setdefault("meta", {})["rephrase_output"] = rephrased != state["question"]
            state.setdefault("meta", {})["rephrased_question"] = rephrased
    except Exception as exc:
        logger.warning("Rephrase LLM call failed: %s, using original question", exc)
        state["rephrased_question"] = state["question"]
        state.setdefault("meta", {})["rephrase_used"] = False
        state.setdefault("meta", {})["rephrase_error"] = str(exc)
    
    return state


def _is_false_positive(query: str, chunks: List[RetrievedChunk]) -> bool:
    if not chunks:
        return False
    q_lower = query.lower()
    
    # Normalizing Turkish characters for robust matching
    q_norm = q_lower.replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c")
    
    keywords = [
        "insansiz kara", "insansız kara", "kara arac",
        "insansiz hava", "insansız hava", "iha",
        "drone", "robolig", "robotik",
        "yazilim", "yazılım",
        "yapay zeka", "saglikta yapay", "sağlıkta yapay",
        "model uydu", "roket", "kuantum",
        "biyoteknoloji", "blokzincir", "cip tasarim", "çip tasarım",
        "e-ticaret", "surdurulebilir sehir", "finansal teknoloji", "fintek",
        "uydu terminal", "hyperloop", "deniz araci", "su alti",
        "jet motor", "iklim degisikligi", "kutup arastirma",
        "nukleer enerji", "onkoloji", "pardus", "robotaksi",
        "tarim", "dogal dil", "elektronik harp", "maden",
        "yol guvenligi", "yol güvenliği", "lojistik",
        "havacilikta yapay", "havacılıkta yapay", "hava savunma",
        "elektrikli arac", "elektrikli araç"
    ]
    
    matched_keywords = []
    for kw in keywords:
        kw_norm = kw.replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c")
        if kw in q_lower or kw_norm in q_norm:
            matched_keywords.append(kw)
            
    if not matched_keywords:
        return False
        
    combined_content = " ".join([c.content.lower() for c in chunks[:3]])
    combined_content_norm = combined_content.replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c")
    
    for kw in matched_keywords:
        kw_norm = kw.replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c")
        if kw in combined_content or kw_norm in combined_content_norm:
            return False # true positive (found keyword in top chunks)
            
    return True # false positive (keyword present in query but not in top chunks)


# ---- Local RAG ----

def node_local_rag(state: GraphState, settings: Settings) -> GraphState:
    vs = build_local_docs_retriever(settings)
    query = normalize_turkish_query(state.get("rephrased_question") or state["question"])
    chunks = retrieve_from_vectorstore(
        vs, query=query,
        source_type="local_docs",
        k=settings.retrieval_top_k,
    )
    
    # Check if the retrieved chunks are a false positive match
    raw_query = state.get("rephrased_question") or state["question"]
    if _is_false_positive(raw_query, chunks):
        logger.info("Local RAG false positive detected for query: %s. Clearing chunks.", raw_query)
        chunks = []
    
    confidence = estimate_chunk_confidence(chunks, settings)
    state.setdefault("meta", {})["local_confidence"] = confidence
    
    MIN_INDIVIDUAL_SCORE = getattr(settings, "rag_hard_floor_score", 0.45)
    MIN_QUALIFYING_CHUNKS = 1
    
    qualifying_chunks = [
        c for c in chunks
        if c.score is not None and max(0.0, min(1.0, 1.0 - (float(c.score) / 2.0))) >= MIN_INDIVIDUAL_SCORE
    ]

    if chunks and confidence >= settings.rag_confidence_threshold:
        state.setdefault("retrieved_chunks", []).extend(chunks)
        state.setdefault("meta", {})["local_rag_reason"] = "confidence_above_threshold"
        return state
        
    elif len(qualifying_chunks) >= MIN_QUALIFYING_CHUNKS:
        state.setdefault("retrieved_chunks", []).extend(qualifying_chunks)
        state.setdefault("meta", {})["local_rag_reason"] = "qualifying_chunks_found"
        state.setdefault("meta", {})["original_confidence"] = confidence
        return state
        
    # Not enough confidence — mark for fallback
    state.setdefault("meta", {})["local_rag_empty"] = True
    state.setdefault("meta", {})["local_rag_reason"] = "fallback_to_site"
    # (Since we didn't add chunks to state yet, we don't need to remove them like before)
    return state


def _decide_after_local_rag(state: GraphState) -> str:
    has_local = any(
        c.source_type == "local_docs" for c in state.get("retrieved_chunks", [])
    )
    # Hem rephrased hem orijinal soruda keyword ara
    question = (state.get("rephrased_question") or state.get("question", "")).lower()
    needs_tavily = _needs_tavily(question)

    if has_local and not needs_tavily:
        return "reranker"
    return "teknofest_web"


# ---- Teknofest site RAG ----

def node_teknofest_web(state: GraphState, settings: Settings) -> GraphState:
    vs = build_teknofest_site_retriever(settings)
    query = normalize_turkish_query(state.get("rephrased_question") or state["question"])
    chunks = retrieve_from_vectorstore(
        vs, query=query,
        source_type="teknofest_site",
        k=settings.retrieval_top_k,
    )
    
    # Check if the retrieved chunks are a false positive match
    raw_query = state.get("rephrased_question") or state["question"]
    if _is_false_positive(raw_query, chunks):
        logger.info("Site RAG false positive detected for query: %s. Clearing chunks.", raw_query)
        chunks = []
        
    state.setdefault("retrieved_chunks", []).extend(chunks)
    state.setdefault("meta", {})["site_confidence"] = estimate_chunk_confidence(chunks, settings)
    if chunks and estimate_chunk_confidence(chunks, settings) >= settings.rag_confidence_threshold:
        return state
    state["meta"]["teknofest_site_empty"] = True
    state["retrieved_chunks"] = [
        c for c in state.get("retrieved_chunks", []) if c.source_type != "teknofest_site"
    ]
    return state


def _decide_after_teknofest_site(state: GraphState) -> str:
    has_site = any(
        c.source_type == "teknofest_site" for c in state.get("retrieved_chunks", [])
    )
    question = (state.get("rephrased_question") or state.get("question", "")).lower()
    if _needs_tavily(question):
        return "tavily_web"
    if has_site:
        return "reranker"
    # No strong site evidence: always try Tavily before LLM knowledge fallback.
    return "tavily_web"


# ---- Tavily web fallback ----

def node_tavily_web(state: GraphState, settings: Settings) -> GraphState:
    try:
        tavily_tool = build_tavily_tool(settings)
        query = normalize_turkish_query(state.get("rephrased_question") or state["question"])
        chunks = retrieve_from_tavily(tavily_tool, query=query)
        state.setdefault("retrieved_chunks", []).extend(chunks)
        state.setdefault("meta", {})["tavily_used"] = bool(chunks)
    except RuntimeError as exc:
        logger.warning("Tavily unavailable: %s", exc)
        state.setdefault("meta", {})["tavily_used"] = False
    return state


def _decide_after_tavily(state: GraphState) -> str:
    """If Tavily produced chunks, proceed to reranker; otherwise fall back to LLM knowledge."""
    if state.get("retrieved_chunks"):
        return "reranker"
    return "llm_knowledge"


# ---- Reranker ----

async def node_reranker(state: GraphState, settings: Settings) -> GraphState:
    retrieved = state.get("retrieved_chunks", [])
    
    # Q9 Expert Advice: Deduplicate by content identity (exact duplicates via content_hash)
    deduped = []
    seen_hashes = set()
    for ch in retrieved:
        h = ch.metadata.get("content_hash")
        if h:
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        deduped.append(ch)
    retrieved = deduped

    if not retrieved:
        state["context_chunks"] = []
        return state

    if settings.reranker_enabled:
        llm = _build_llm(settings, temperature=0.0, purpose="reranker")
        final = await rerank_chunks(
            query=normalize_turkish_query(state.get("rephrased_question") or state["question"]),
            chunks=retrieved,
            llm=llm,
            final_k=settings.retrieval_final_k,
        )
        state.setdefault("meta", {})["reranker_used"] = True
    else:
        # No reranking — just take the top final_k by insertion order
        final = retrieved[: settings.retrieval_final_k]
        state.setdefault("meta", {})["reranker_used"] = False

    state["context_chunks"] = final
    return state


# ---- Context builder ----

def node_context_builder(state: GraphState, settings: Settings) -> GraphState:
    chunks = state.get("context_chunks") or state.get("retrieved_chunks", [])
    context_str, selected = build_context(
        chunks,
        max_total_chars=12_000,
        min_score=None,   # score filtering already done via confidence threshold
        min_rerank=None,  # rerank threshold left to the reranker node
    )
    state["context_chunks"] = selected
    state["context_str"] = context_str
    state.setdefault("meta", {})["context_chunks_count"] = len(selected)
    return state


# ---- Freshness gate ----

@traceable(
    run_type="chain", 
    name="freshness_gate",
    metadata={"pipeline_stage": "routing"}
)
def node_freshness_gate(state: GraphState, settings: Settings) -> GraphState:
    """
    Inspects context_chunks AFTER context_builder.
    Sets state["meta"]["freshness_decision"] to:
      - "serve_directly"  — chunks are trusted and query is not time-sensitive
      - "fetch_live"      — query is time-sensitive OR chunks are from a stale source
    """
    question = (
        state.get("rephrased_question") or state.get("question", "")
    ).lower()

    # Check 1 — Is the query about time-sensitive data?
    is_sensitive = any(kw in question for kw in FRESHNESS_SENSITIVE_KEYWORDS)

    # Check 2 — Are any context chunks from a potentially stale source?
    chunks = state.get("context_chunks", [])
    stale_chunks = [
        c for c in chunks
        if any(marker in c.source for marker in STALE_SOURCE_MARKERS)
    ]
    has_stale = bool(stale_chunks)

    decision = "fetch_live" if (is_sensitive or has_stale) else "serve_directly"

    try:
        from langsmith import get_current_run_tree
        run = get_current_run_tree()
        if run:
            run.add_outputs({
                "decision": decision,
                "reason": f"Sensitive: {is_sensitive}, Stale chunks: {len(stale_chunks)}"
            })
    except Exception:
        pass

    state.setdefault("meta", {})["freshness_decision"] = decision
    state.setdefault("meta", {})["freshness_stale_sources"] = [
        c.source for c in stale_chunks
    ]
    state.setdefault("meta", {})["freshness_is_sensitive"] = is_sensitive
    logger.info(
        "Freshness gate: sensitive=%s stale_chunks=%d → %s",
        is_sensitive, len(stale_chunks), decision,
    )
    return state


def _decide_after_freshness_gate(state: GraphState) -> str:
    return state.get("meta", {}).get("freshness_decision", "serve_directly")


# ---- Live fetch nodes ----

@traceable(run_type="tool", name="live_page_fetch", metadata={"pipeline_stage": "live_fetch"})
async def node_live_page_fetch(state: GraphState, settings: Settings) -> GraphState:
    from app.agent.tools import fetch_teknofest_competition_page
    
    question = state.get("rephrased_question") or state["question"]
    lower_q = question.lower()
    
    slug_map = {
        # --- Short keywords & Shorthands ---
        "insansiz kara": "insansiz-kara-araci-yarismasi",
        "insansız kara": "insansiz-kara-araci-yarismasi",
        "kara arac": "insansiz-kara-araci-yarismasi",
        "insansiz hava": "insansiz-hava-araci-yarismasi",
        "insansız hava": "insansiz-hava-araci-yarismasi",
        "iha": "insansiz-hava-araci-yarismasi",
        "drone": "teknofest-drone-sampiyonasi",
        "robolig": "teknofest-robolig-yarismasi",
        "robotik": "sanayide-robotik-uygulamalar-yarismasi",
        "yazilim": "teknofest-yazilim-yarismasi",
        "yazılım": "teknofest-yazilim-yarismasi",
        "yapay zeka": "saglikta-yapay-zeka-yarismasi",
        "saglikta yapay": "saglikta-yapay-zeka-yarismasi",
        "sağlıkta yapay": "saglikta-yapay-zeka-yarismasi",
        "model uydu": "model-uydu-yarismasi",
        "roket": "roket-yarismasi",
        "kuantum": "kuantum-teknolojileri-yarismasi",
        "biyoteknoloji": "biyoteknoloji-inovasyon-yarismasi",
        "blokzincir": "blokzincir-yarismasi",
        "cip tasarim": "cip-tasarim-yarismasi",
        "çip tasarım": "cip-tasarim-yarismasi",
        "e-ticaret": "e-ticaret-yarismasi",
        "surdurulebilir sehir": "gelecegin-surdurulebilir-sehirleri-yarismasi",
        "finansal teknoloji": "finansal-teknolojiler-yarismasi",
        "fintek": "finansal-teknolojiler-yarismasi",
        "uydu terminal": "hareketli-uydu-terminali-yarismasi",
        "hyperloop": "hyperloop-gelistirme-yarismasi",
        "deniz araci": "insansiz-deniz-araci-yarismasi",
        "su alti": "insansiz-su-alti-sistemleri-yarismasi",
        "jet motor": "jet-motor-tasarim-yarismasi",
        "iklim degisikligi": "lise-ogrencileri-iklim-degisikligi-arastirma-projeleri-yarismasi",
        "kutup arastirma": "lise-ogrencileri-kutup-arastirma-projeleri-yarismasi",
        "nukleer enerji": "nukleer-enerji-teknolojileri-tasarim-yarismasi",
        "onkoloji": "onkolojide-3t-yarismasi",
        "pardus": "pardus-hata-yakalama-ve-oneri-yarismasi",
        "robotaksi": "robotaksi-binek-otonom-arac-yarismasi",
        "tarim": "tarim-teknolojileri-yarismasi",
        "dogal dil": "turkce-dogal-dil-isleme-yarismasi",
        "elektronik harp": "elektronik-harp-yarismasi",
        "maden": "maden-teknolojileri-yarismasi",
        
        # --- Full Category Names ---
        "5g & yapay zeka ile akıllı yol güvenliği yarışması": "5g-yapay-zeka-ile-akilli-yol-guvenligi-yarismasi",
        "biyoteknoloji i̇novasyon yarışması": "biyoteknoloji-inovasyon-yarismasi",
        "blokzincir yarışması": "blokzincir-yarismasi",
        "çip tasarım yarışması": "cip-tasarim-yarismasi",
        "dikey i̇nişli roket yarışması": "dikey-inisli-roket-yarismasi",
        "e-ticaret yarışması": "e-ticaret-yarismasi",
        "geleceğin sürdürülebilir şehirleri yarışması": "gelecegin-surdurulebilir-sehirleri-yarismasi",
        "yapay zeka destekli lojistik anahat optimizasyonu yarışması": "yapay-zeka-destekli-lojistik-anahat-optimizasyonu-yarismasi",
        "finansal teknolojiler yarışması": "finansal-teknolojiler-yarismasi",
        "hareketli uydu terminali yarışması": "hareketli-uydu-terminali-yarismasi",
        "havacılıkta yapay zeka yarışması": "havacilikta-yapay-zeka-yarismasi",
        "çelikkubbe hava savunma sistemleri yarışması": "celikkubbe-hava-savunma-sistemleri-yarismasi",
        "hyperloop geliştirme yarışması": "hyperloop-gelistirme-yarismasi",
        "i̇nsansız deniz aracı yarışması": "insansiz-deniz-araci-yarismasi",
        "i̇nsansız kara aracı yarışması": "insansiz-kara-araci-yarismasi",
        "i̇nsansız su altı sistemleri yarışması": "insansiz-su-alti-sistemleri-yarismasi",
        "i̇nsansız su altı sistemleri yıldızlar yarışması": "insansiz-su-alti-sistemleri-yildizlar-yarismasi",
        "jet motor tasarım yarışması": "jet-motor-tasarim-yarismasi",
        "kuantum teknolojileri yarışması": "kuantum-teknolojileri-yarismasi",
        "liseler arası i̇nsansız hava araçları yarışması": "liseler-arasi-insansiz-hava-araclari-yarismasi",
        "lise öğrencileri i̇klim değişikliği araştırma projeleri yarışması": "lise-ogrencileri-iklim-degisikligi-arastirma-projeleri-yarismasi",
        "lise öğrencileri kutup araştırma projeleri yarışması": "lise-ogrencileri-kutup-arastirma-projeleri-yarismasi",
        "model uydu yarışması": "model-uydu-yarismasi",
        "nükleer enerji teknolojileri tasarım yarışması": "nukleer-enerji-teknolojileri-tasarim-yarismasi",
        "onkolojide 3t yarışması": "onkolojide-3t-yarismasi",
        "pardus hata yakalama ve öneri yarışması": "pardus-hata-yakalama-ve-oneri-yarismasi",
        "robotaksi-binek otonom araç yarışması": "robotaksi-binek-otonom-arac-yarismasi",
        "roket yarışması": "roket-yarismasi",
        "sağlıkta yapay zeka yarışması": "saglikta-yapay-zeka-yarismasi",
        "sanayide robotik uygulamalar yarışması": "sanayide-robotik-uygulamalar-yarismasi",
        "sürü i̇ha yarışması": "suru-iha-yarismasi",
        "savaşan i̇ha yıldızlar yarışması": "savasan-iha-yildizlar-yarismasi",
        "savaşan i̇ha yarışması": "savasan-iha-yarismasi",
        "savaşan i̇ha avcı drone yarışması": "savasan-iha-avci-drone-yarismasi",
        "su altı roket yarışması": "su-alti-roket-yarismasi",
        "tarım teknolojileri yarışması": "tarim-teknolojileri-yarismasi",
        "teknofest drone şampiyonası": "teknofest-drone-sampiyonasi",
        "teknofest mimari ve görsel tasarım yarışması": "teknofest-mimari-ve-gorsel-tasarim-yarismasi",
        "teknofest robolig yarışması": "teknofest-robolig-yarismasi",
        "world drone cup": "world-drone-cup",
        "uluslararası elektrikli araç yarışları": "uluslararasi-elektrikli-arac-yarislari",
        "uluslararası i̇nsansız hava aracı yarışması": "uluslararasi-insansiz-hava-araci-yarismasi",
        "türkçe doğal dil i̇şleme yarışması": "turkce-dogal-dil-isleme-yarismasi",
        "yapay zeka destekli havayolu optimizasyonu yarışması": "yapay-zeka-destekli-havayolu-optimizasyonu-yarismasi",
        "elektronik harp yarışması": "elektronik-harp-yarismasi",
        "maden teknolojileri yarışması": "maden-teknolojileri-yarismasi",
        "i̇leri otonom sistemler tasarım ve operasyon yarışması": "ileri-otonom-sistemler-tasarim-ve-operasyon-yarismasi",
    }
    slug = None
    for keyword, candidate_slug in slug_map.items():
        if keyword in lower_q:
            slug = candidate_slug
            break

    live_page_content = ""
    url_attempted = ""
    if slug:
        url_attempted = f"https://www.teknofest.org/tr/yarismalar/{slug}/"
        try:
            live_page_content = fetch_teknofest_competition_page.invoke({"competition_slug": slug})
            if live_page_content.startswith("[FETCH ERROR]") or len(live_page_content) < 200:
                live_page_content = ""
                logger.warning("Live fetch Step 1 returned error/empty for slug '%s'", slug)
            else:
                logger.info("Live fetch Step 1 OK: %d chars from %s", len(live_page_content), slug)
        except Exception as exc:
            logger.warning("Live fetch Step 1 exception: %s", exc)
            live_page_content = ""

    state.setdefault("meta", {})["live_fetch_step1_slug"] = slug
    state.setdefault("meta", {})["live_fetch_step1_ok"] = bool(live_page_content)

    if live_page_content and slug:
        from app.rag.retrievers import RetrievedChunk
        live_chunk = RetrievedChunk(
            content=live_page_content[:10000],
            metadata={
                "url": url_attempted,
                "source": url_attempted,
                "title": "TEKNOFEST Yarışma Sayfası (Canlı)",
                "document_type": "web_live",
                "source_priority": 0,
                "fetched_at": __import__('datetime').datetime.now().isoformat(),
            },
            score=0.0,
            source_type="teknofest_site",
        )
        state["context_chunks"] = [live_chunk]
        state.setdefault("meta", {})["live_fetch_source"] = "teknofest.org (live)"
    
    try:
        from langsmith import get_current_run_tree
        run = get_current_run_tree()
        if run:
            run.add_outputs({
                "url_attempted": url_attempted,
                "success": bool(live_page_content),
                "content_length": len(live_page_content)
            })
    except Exception:
        pass

    return state

def _decide_after_live_page_fetch(state: GraphState) -> str:
    if state.get("meta", {}).get("live_fetch_step1_ok"):
        return "live_answer_synthesizer"
    return "live_tavily_fallback"

@traceable(run_type="tool", name="live_tavily_fallback", metadata={"pipeline_stage": "live_fetch"})
async def node_live_tavily_fallback(state: GraphState, settings: Settings) -> GraphState:
    question = state.get("rephrased_question") or state["question"]
    tavily_content = ""
    try:
        tavily_tool = build_tavily_tool(settings)
        tavily_query = f"{question} 2026 site:teknofest.org"
        chunks = retrieve_from_tavily(tavily_tool, query=tavily_query)
        if not chunks:
            chunks = retrieve_from_tavily(tavily_tool, query=f"TEKNOFEST 2026 {question}")
        if chunks:
            context_str, _ = build_context(chunks, max_total_chars=4_000)
            tavily_content = context_str
            state.setdefault("retrieved_chunks", []).extend(chunks)
            state["context_chunks"] = chunks
            state.setdefault("meta", {})["live_fetch_source"] = "tavily"
            logger.info("Live fetch Step 2 (Tavily) OK: %d chunks", len(chunks))
    except Exception as exc:
        logger.warning("Live fetch Step 2 (Tavily) failed: %s", exc)

    state.setdefault("meta", {})["live_fetch_step2_ok"] = bool(tavily_content)
    return state

@traceable(run_type="llm", name="live_answer_synthesizer", metadata={"pipeline_stage": "response_generation"})
async def node_live_answer_synthesizer(state: GraphState, settings: Settings) -> GraphState:
    question = state.get("rephrased_question") or state["question"]
    chunks = state.get("context_chunks", [])
    combined_context, _ = build_context(chunks, max_total_chars=10_000, min_score=None)
    
    llm = _build_llm(settings, temperature=0.1, purpose="main")

    if combined_context:
        sys_prompt = (
            PERSONALITY_PREFIX + "\n\n"
            "Sen TEKNOFEST bilgi asistanısın.\n"
            "Aşağıda TEKNOFEST'in resmi sayfasından veya güvenilir web kaynaklarından alınmış CANLI veri var.\n"
            "Bu veriden gelen rakamları (ödül miktarı, tarih vb.) kullanabilirsin — bunlar gerçek 2026 verisidir.\n"
            "Eğer soru ödül miktarı, tarih veya sayısal bir değer gibi zaman hassasiyeti olan bir bilgi istiyorsa ve bağlamda rakam/tarih YOKsa: 'Bu konuda güncel bilgiye ulaşamadım.' de. Genel tanıtım veya açıklama sorularında bağlamdaki açıklamalara göre detaylı bilgi ver.\n"
            "ASLA bağlamda olmayan bir rakam üretme — ne eski VectorDB verisini ne de kendi hafızanı kullan.\n"
            "YASAK: 'ziyaret edin', 'kontrol edin', 'resmi siteye bakın' gibi yönlendirme ifadeleri.\n"
            "Türkçe cevapla."
        )
        user_prompt = (
            f"Canlı kaynak verisi:\n{combined_context}\n\n"
            f"Kullanıcı sorusu: {question}\n\n"
            "Yalnızca yukarıdaki canlı veriye dayanarak cevapla."
        )
    else:
        sys_prompt = (
            PERSONALITY_PREFIX + "\n\n"
            "Sen TEKNOFEST bilgi asistanısın.\n"
            "Bu soru için canlı veriye ulaşamadın. Bunu dürüstçe belirt.\n"
            "ASLA ödül miktarı, tarih veya sıralama üretme.\n"
            "Türkçe cevapla."
        )
        user_prompt = (
            f"Soru: {question}\n\n"
            "Canlı veri mevcut değil. Dürüstçe bilmediğini belirt; "
            "kullanıcıya 'teknofest.org/tr/yarismalar sayfasını inceleyebilirsin' de."
        )

    try:
        res = await llm.ainvoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], config={"tags": ["final_synthesizer"]})
        answer = (res.content or "").strip()
    except Exception as exc:
        logger.error("Live fetch Step 3 LLM failed: %s", exc)
        answer = "Şu an teknik bir sorun yaşıyorum, lütfen tekrar dene."

    state["answer"] = answer
    state["route_taken"] = "live_fetch"
    
    source_label = state.get("meta", {}).get("live_fetch_source", "none")
    try:
        from langsmith import get_current_run_tree
        run = get_current_run_tree()
        if run:
            run.add_outputs({
                "source_label": source_label,
                "final_answer": answer
            })
    except Exception:
        pass

    return state


# ---- Direct LLM (non-TEKNOFEST) ----

async def node_direct_llm(state: GraphState, settings: Settings) -> GraphState:
    state["answer"] = "Bu konuda yardımcı olamıyorum ama TEKNOFEST hakkında her şeyi sorabilirsin! Yarışmalar veya etkinliklerle ilgili merak ettiğin bir şey var mı?"
    state["route_taken"] = "direct"
    state["context_chunks"] = []
    state["retrieved_chunks"] = []
    return state

# ---- Kisisel LLM ----

async def node_kisisel_llm(state: GraphState, settings: Settings) -> GraphState:
    llm_gen = _build_llm(settings, temperature=0.7, purpose="main")
    category_context = _extract_category_context(
        state.get("chat_history") or [],
        state.get("rephrased_question") or state.get("question", ""),
    )
    if category_context:
        category_hint = f"Konuşma geçmişinden tespit edilen kategori: {category_context}. Cevabında bu kategori adını aynen kullan."
    else:
        category_hint = "Konuşma geçmişinde net bir kategori adı tespit edilemedi; kategori varsa kullanıcıdan gelen bağlamı koru."
    sys_prompt = PERSONALITY_PREFIX + f"\n\n{category_hint}\nSen TEKNOFEST'in sıcak ve samimi chatbot'usun. Sana kişisel bir soru soruldu. Eğlenceli ve destekleyici bir şekilde cevap ver, sonra konuyu nazikçe TEKNOFEST'e çek. 'Senin kategorin' gibi belirsiz ifadeler yerine tespit edilen kategori adını kullan."
    ans_res = await llm_gen.ainvoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": state["question"]}
    ], config={"tags": ["final_synthesizer"]})
    state["answer"] = ans_res.content
    state["route_taken"] = "kisisel"
    state["context_chunks"] = []
    state["retrieved_chunks"] = []
    return state


# ---- LLM Knowledge Fallback ----

async def node_llm_knowledge(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.3, purpose="main")
    question = state.get("rephrased_question") or state["question"]

    # Last-chance live retrieval: retry Tavily here before giving a generic answer.
    live_chunks: List[RetrievedChunk] = []
    try:
        tavily_tool = build_tavily_tool(settings)
        normalized_question = normalize_turkish_query(question)
        live_chunks = retrieve_from_tavily(tavily_tool, query=normalized_question)
        if not live_chunks:
            enriched_query = f"TEKNOFEST {normalized_question}"
            live_chunks = retrieve_from_tavily(tavily_tool, query=enriched_query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM knowledge Tavily retry failed: %s", exc)

    answer_text = ""
    if live_chunks:
        context_str, selected = build_context(live_chunks, max_total_chars=4_000)
        state["context_chunks"] = selected
        state.setdefault("retrieved_chunks", []).extend(live_chunks)
        sys_prompt = (
            PERSONALITY_PREFIX + "\n\n"
            "Sen TEKNOFEST uzmanısın. Sana Tavily'den gerçek kaynaklar sağlandıysa onları kullanarak somut bilgi ver.\n"
            "Kaynaklarda geçen rakamları (ödül miktarı, tarih vb.) kullanabilirsin — bunlar gerçek verilerdir.\n"
            "Eğer soru ödül miktarı, tarih veya sayısal bir değer gibi zaman hassasiyeti olan bir bilgi istiyorsa ve kaynaklarda rakam/tarih YOKsa: 'Bu konuda elimde yeterli bilgi bulunmuyor' de. Genel tanıtım veya açıklama sorularında kaynaklardaki açıklamalara göre detaylı bilgi ver.\n"
            "Asla kaynaklarda olmayan bir rakam üretme.\n"
            "YASAK: Kullanıcıyı dış siteye yönlendirme, 'ziyaret edin', 'kontrol edin', 'resmi siteye bakın' gibi ifadeler kullanma.\n"
            "Cevabını yapısal olarak organize et: başlıklar, madde listeleri veya tablolar kullan.\n"
            "Bağlamdaki TÜM ilgili detayları ver — tek paragraflık belirsiz bir cevap yerine kapsamlı bir cevap ver.\n"
            "Tarih, yer, ödül miktarı, kategori gibi somut veriler varsa mutlaka belirt.\n"
            "Cevabın sonunda her zaman kaynak bilgisini belirt.\n"
            "Türkçe cevapla."
        )
        user_prompt = (
            f"Canlı bağlam:\n{context_str}\n\n"
            f"Soru: {question}\n\n"
            "Yalnızca bağlam destekliyorsa sayı/tarih belirt."
        )
        res = await llm.ainvoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], config={"tags": ["final_synthesizer"]})
        answer_text = (res.content or "").strip()
    else:
        sys_prompt = (
            PERSONALITY_PREFIX + "\n\n"
            "Sen TEKNOFEST uzmanısın. Sana Tavily'den gerçek kaynaklar sağlandıysa onları kullanarak somut bilgi ver.\n"
            "Kaynaklarda geçen rakamları (ödül miktarı, tarih vb.) kullanabilirsin — bunlar gerçek verilerdir.\n"
            "Eğer soru ödül miktarı, tarih veya sayısal bir değer gibi zaman hassasiyeti olan bir bilgi istiyorsa ve kaynaklarda rakam/tarih YOKsa: 'Bu konuda elimde yeterli bilgi bulunmuyor' de. Genel tanıtım veya açıklama sorularında kaynaklardaki açıklamalara göre detaylı bilgi ver.\n"
            "Asla kaynaklarda olmayan bir rakam üretme.\n"
            "YASAK: Kullanıcıyı dış siteye yönlendirme, 'ziyaret edin', 'kontrol edin', 'resmi siteye bakın' gibi ifadeler kullanma.\n"
            "Cevabını yapısal olarak organize et: başlıklar, madde listeleri veya tablolar kullan.\n"
            "Bağlamdaki TÜM ilgili detayları ver — tek paragraflık belirsiz bir cevap yerine kapsamlı bir cevap ver.\n"
            "Tarih, yer, ödül miktarı, kategori gibi somut veriler varsa mutlaka belirt.\n"
            "Cevabın sonunda her zaman kaynak bilgisini belirt.\n"
            "Türkçe cevapla."
        )
        res = await llm.ainvoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ], config={"tags": ["final_synthesizer"]})
        answer_text = (res.content or "").strip()

    if _contains_disallowed_redirect(answer_text):
        rewrite_prompt = (
            "Aşağıdaki cevabı, kullanıcıyı başka kaynağa yönlendirmeden yeniden yaz. "
            "Somut bilgiyi koru, 'kontrol edin/ziyaret edin/resmi kaynak' ifadelerini tamamen kaldır.\n\n"
            f"Cevap:\n{answer_text}"
        )
        rewrite = await llm.ainvoke([{"role": "user", "content": rewrite_prompt}])
        answer_text = (rewrite.content or answer_text).strip()

    state["answer"] = answer_text
    state["route_taken"] = "llm_knowledge"
    if not live_chunks:
        state["context_chunks"] = []
        state["retrieved_chunks"] = []
    return state


# ---- Answer synthesizer (F.3 — Agent entegrasyonu) ----

async def node_answer_synthesizer(state: GraphState, settings: Settings) -> GraphState:
    """
    F.3 — RAG bağlamını agent'e geçirir; agent tool calling loop ile
    gerekirse gerçek zamanlı bilgi çekerek cevabı üretir.
    route_taken ve prompt_preview mantığı korunur.

    NOTE: This node is only reached when freshness_gate decided "serve_directly".
    For time-sensitive queries with stale sources, `node_live_fetch` is used instead.
    """
    from app.agent.agent_node import run_agent_node  # lazy import — döngüsel import önleme

    context_block = state.get("context_str") or ""
    if not context_block:
        context_block, _ = build_context(state.get("context_chunks", []))

    question = state.get("rephrased_question") or state["question"]

    # Prompt preview — LangSmith trace için
    prompt_preview = f"Soru: {question}\nBağlam (ilk 400): {context_block[:400]}"
    state.setdefault("meta", {})["prompt_preview"] = prompt_preview

    # F.3 — Agent node çağrısı (tool calling loop içeriyor)
    answer = await run_agent_node(
        question=question,
        context_str=context_block,
        settings=settings,
        chat_history=state.get("chat_history"),
    )
    state["answer"] = answer

    # Determine route label from dominant source type (mevcut mantık korunur)
    types = {ch.source_type for ch in state.get("context_chunks", [])}
    route: RouteLiteral = "local"
    if "teknofest_site" in types:
        route = "site"
    if "tavily" in types and len(types) == 1:
        route = "tavily"
    state["route_taken"] = route
    return state



# ---- Hallucination guard ----

async def node_hallucination_guard(state: GraphState, settings: Settings) -> GraphState:
    answer = state.get("answer", "")
    route_taken = state.get("route_taken", "")
    
    # Only flag if: no real sources AND answer contains invented numbers AND doesn't cite teknofest
    if len(state.get("context_chunks", [])) == 0 and "teknofest.org" not in answer.lower() and any(
        keyword in answer for keyword in ["TL", "milyon", "bin TL"]
    ):
        state["answer"] = "Bu konuda elimde doğrulanmış bilgi bulunmuyor. Lütfen soruyu daha detaylı sorarak tekrar dene; sana daha iyi yardımcı olayım."
        state.setdefault("meta", {})["hallucination_detected"] = True
        hal_status = "suspicious"
    else:
        result = await hallucination_check(
            settings=settings,
            question=state["question"],
            answer=answer,
            context_chunks=state.get("context_chunks", []),
        )
        state.setdefault("meta", {})["hallucination_check"] = result
        hal_status = result.get("status", "unknown")

        # Routes that generate answers without RAG context — never override them
        # live_fetch answers are grounded in real-time data: treat as bypass
        bypass_routes = {"llm_knowledge", "kisisel", "direct", "live_fetch"}
        current_route = state.get("route_taken", "")

        # Allow the agent's own actionable rejection or tool-fetched answer to pass through
        if current_route not in bypass_routes and not state.get("context_chunks"):
            pass

    # --- Evaluation logging ---
    retrieved = state.get("retrieved_chunks", [])
    selected = state.get("context_chunks", [])
    _log_eval(settings, state, retrieved, selected, hal_status)

    return state


def _decide_after_hallucination_guard(state: GraphState) -> str:
    """
    If agent rejected answer but we have context (or generic fallback is needed),
    route to llm_knowledge for general knowledge synthesis.
    """
    answer = state.get("answer", "").strip()
    has_context = bool(state.get("context_chunks"))
    
    answer_lower = answer.lower()
    rejection_keywords = [
        "bulunamadı", "bulamadım", "ulaşamadım", "erişemedim", 
        "bilgi bulunmuyor", "bilgi yok", "bilgiye sahip değilim", 
        "bilgi mevcut değil", "bilgiye ulaşamadım", "bilgi bulunmamaktadır",
        "yeterli bilgi", "ulaşılamadı", "doğrulanmış bilgi"
    ]
    is_rejection = (
        any(kw in answer_lower for kw in rejection_keywords)
        or answer == "Sağlanan belgelerden ilgili bilgi bulamadım."
    )
    current_route = state.get("route_taken", "")
    
    # If we have context but agent said "not found", fallback to general knowledge
    if is_rejection and current_route in ("local", "site", "tavily"):
        return "llm_knowledge"
    
    # Otherwise just end
    return "end"


def _log_eval(
    settings: Settings,
    state: GraphState,
    retrieved: List[RetrievedChunk],
    selected: List[RetrievedChunk],
    hal_status: str,
) -> None:
    """Fire-and-forget evaluation log write + LangSmith metadata injection."""
    try:
        failure_tags = tag_failures(
            retrieved_chunks=retrieved,
            answer=state.get("answer", ""),
            hallucination_status=hal_status,
            recall=1.0,  # No ground-truth at runtime; avoid auto-tagging
        )
        log_retrieval_event(
            log_path=settings.eval_log_path,
            query=state.get("question", ""),
            retrieved_chunks=retrieved,
            selected_chunks=selected,
            answer=state.get("answer", ""),
            route=state.get("route_taken", "unknown"),
            hallucination_status=hal_status,
            failure_tags=failure_tags,
            # D.5 — rephrase observability
            rephrased_question=state.get("rephrased_question") or None,
            chat_history_length=len(state.get("chat_history", [])),
        )

        # Attach rich metadata to the active LangSmith run so it's visible
        # in the run metadata panel without any extra API calls.
        if is_tracing_enabled():
            try:
                from langsmith import get_current_run_tree  # lazy import
                run = get_current_run_tree()
                if run is not None:
                    run.add_metadata(
                        get_run_metadata(
                            retrieved_chunks=retrieved,
                            selected_chunks=selected,
                            prompt_preview=state.get("meta", {}).get("prompt_preview"),
                            extra={
                                "route": state.get("route_taken", "unknown"),
                                "hallucination_status": hal_status,
                                "failure_tags": failure_tags,
                                "reranker_used": state.get("meta", {}).get("reranker_used"),
                                "intent": state.get("intent"),
                            },
                        )
                    )
            except Exception:  # noqa: BLE001
                pass  # never let tracing crash the pipeline

    except Exception as exc:  # noqa: BLE001
        logger.warning("Eval log failed: %s", exc)



# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_teknofest_graph(settings: Settings):
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # ---- Wrap nodes with settings closure ----

    async def intent_node(s: GraphState) -> GraphState:
        return await node_intent_classification(s, settings)

    async def rephrase_node(s: GraphState) -> GraphState:          # D.3
        return await node_rephrase(s, settings)

    def local_rag_node(s: GraphState) -> GraphState:
        return node_local_rag(s, settings)

    def teknofest_web_node(s: GraphState) -> GraphState:
        return node_teknofest_web(s, settings)

    def tavily_web_node(s: GraphState) -> GraphState:
        return node_tavily_web(s, settings)

    async def reranker_node(s: GraphState) -> GraphState:
        return await node_reranker(s, settings)

    def context_builder_node(s: GraphState) -> GraphState:
        return node_context_builder(s, settings)

    async def direct_llm_node(s: GraphState) -> GraphState:
        return await node_direct_llm(s, settings)

    async def kisisel_llm_node(s: GraphState) -> GraphState:
        return await node_kisisel_llm(s, settings)

    async def answer_synthesizer_node(s: GraphState) -> GraphState:
        return await node_answer_synthesizer(s, settings)

    async def llm_knowledge_node(s: GraphState) -> GraphState:
        return await node_llm_knowledge(s, settings)

    async def hallucination_guard_node(s: GraphState) -> GraphState:
        return await node_hallucination_guard(s, settings)

    # ---- Wrap freshness gate & live fetch nodes ----

    def freshness_gate_node(s: GraphState) -> GraphState:
        return node_freshness_gate(s, settings)

    async def live_page_fetch_node(s: GraphState) -> GraphState:
        return await node_live_page_fetch(s, settings)
        
    async def live_tavily_fallback_node(s: GraphState) -> GraphState:
        return await node_live_tavily_fallback(s, settings)
        
    async def live_answer_synthesizer_node(s: GraphState) -> GraphState:
        return await node_live_answer_synthesizer(s, settings)

    # ---- Register nodes ----
    workflow.add_node("intent", intent_node)
    workflow.add_node("rephrase", rephrase_node)                   # D.3
    workflow.add_node("local_rag", local_rag_node)
    workflow.add_node("teknofest_web", teknofest_web_node)
    workflow.add_node("tavily_web", tavily_web_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("freshness_gate", freshness_gate_node)       # NEW
    workflow.add_node("live_page_fetch", live_page_fetch_node)
    workflow.add_node("live_tavily_fallback", live_tavily_fallback_node)
    workflow.add_node("live_answer_synthesizer", live_answer_synthesizer_node)
    workflow.add_node("direct_llm", direct_llm_node)
    workflow.add_node("kisisel_llm", kisisel_llm_node)
    workflow.add_node("llm_knowledge", llm_knowledge_node)
    workflow.add_node("answer_synthesizer", answer_synthesizer_node)
    workflow.add_node("hallucination_guard", hallucination_guard_node)

    # ---- Edges ----
    # rephrase -> intent → conditional (local_rag veya direct_llm)
    workflow.set_entry_point("rephrase")
    workflow.add_edge("rephrase", "intent")
    workflow.add_conditional_edges("intent", _decide_next_after_intent, {
        "local_rag": "local_rag",
        "kisisel_llm": "kisisel_llm",
        "direct_llm": "direct_llm",
    })
    workflow.add_conditional_edges("local_rag", _decide_after_local_rag, {
        "reranker": "reranker",
        "teknofest_web": "teknofest_web",
    })
    workflow.add_conditional_edges("teknofest_web", _decide_after_teknofest_site, {
        "reranker": "reranker",
        "tavily_web": "tavily_web",
        "llm_knowledge": "llm_knowledge",
    })
    workflow.add_conditional_edges("tavily_web", _decide_after_tavily, {
        "reranker": "reranker",
        "llm_knowledge": "llm_knowledge",
    })
    workflow.add_edge("reranker", "context_builder")
    # context_builder → freshness_gate → (serve_directly | fetch_live)
    workflow.add_edge("context_builder", "freshness_gate")         # CHANGED
    workflow.add_conditional_edges("freshness_gate", _decide_after_freshness_gate, {
        "serve_directly": "answer_synthesizer",
        "fetch_live": "live_page_fetch",
    })
    
    workflow.add_conditional_edges("live_page_fetch", _decide_after_live_page_fetch, {
        "live_answer_synthesizer": "live_answer_synthesizer",
        "live_tavily_fallback": "live_tavily_fallback",
    })
    workflow.add_edge("live_tavily_fallback", "live_answer_synthesizer")
    workflow.add_edge("live_answer_synthesizer", "hallucination_guard")
    workflow.add_edge("answer_synthesizer", "hallucination_guard")
    workflow.add_conditional_edges("hallucination_guard", _decide_after_hallucination_guard, {
        "llm_knowledge": "llm_knowledge",
        "end": END,
    })
    workflow.add_edge("llm_knowledge", END)
    workflow.add_edge("direct_llm", END)
    workflow.add_edge("kisisel_llm", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public run helper (Remote Dev Server & Local Fallback)
# ---------------------------------------------------------------------------

import os
import uuid
import httpx
from langgraph_sdk import get_client

def _build_sources_list(context_chunks: List[Any]) -> List[Dict[str, Any]]:
    sources_list = []
    seen_sources = set()
    for ch in context_chunks or []:
        # Support both RetrievedChunk object and serialized dict
        if isinstance(ch, dict):
            metadata = ch.get("metadata") or {}
            source_type = ch.get("source_type") or "local_docs"
            score = ch.get("score")
            content = ch.get("content") or ""
        else:
            metadata = ch.metadata or {}
            source_type = ch.source_type
            score = ch.score
            content = getattr(ch, "content", "")

        src_id = (
            metadata.get("source")
            or metadata.get("url")
            or metadata.get("crawl_source")
            or ""
        )
        if src_id and src_id in seen_sources:
            continue
        if src_id:
            seen_sources.add(src_id)

        source_entry = {
            "type": source_type,
            "metadata": metadata,
            "score": round(score, 3) if score is not None else None,
            "content_preview": content[:200],
        }

        # Güvenilirlik etiketleri
        if source_type == "local_docs":
            source_entry["trust_level"] = "high"
            source_entry["trust_label"] = "Resmi Doküman (Yerel)"
        elif source_type == "teknofest_site":
            source_entry["trust_level"] = "high"
            source_entry["trust_label"] = "TEKNOFEST Resmi Sitesi"
        elif source_type == "tavily":
            url = metadata.get("url", "")
            if any(d in url for d in ["teknofest.org", "cdn.teknofest.org", ".gov.tr"]):
                source_entry["trust_level"] = "high"
                source_entry["trust_label"] = "Resmi Kaynak"
            else:
                source_entry["trust_level"] = "medium"
                source_entry["trust_label"] = "Web Kaynağı"

        sources_list.append(source_entry)
    return sources_list


async def _get_remote_client_url() -> str | None:
    # 1. Check environment variable
    env_url = os.getenv("LANGGRAPH_API_URL")
    if env_url:
        return env_url
    
    # 2. Check default dev server ports
    for port in [2024, 8123]:
        url = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(timeout=0.1) as client:
                res = await client.get(f"{url}/info")
                if res.status_code == 200:
                    return url
        except Exception:
            continue
    return None


@traceable(run_type="chain", name="teknofest-rag-query")
async def run_graph(
    graph,
    question: str,
    chat_history: List[Dict[str, str]] = None,
    callbacks: list | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Wrapper used by FastAPI and tests. Runs on remote dev server if active, otherwise runs locally."""
    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or [],
        "rephrased_question": "",
    }

    # Try running on remote langgraph dev server
    remote_url = await _get_remote_client_url()
    if remote_url:
        try:
            client = get_client(url=remote_url)
            thread_id = (metadata or {}).get("session_id") or str(uuid.uuid4())
            
            try:
                await client.threads.get(thread_id)
            except Exception:
                await client.threads.create(thread_id=thread_id)
                
            run_res = await client.runs.wait(
                thread_id=thread_id,
                assistant_id="teknofest_rag",
                input=initial_state,
            )
            final_state = run_res.get("values", {})
            sources_list = _build_sources_list(final_state.get("context_chunks", []))
            return {
                "answer": final_state.get("answer", ""),
                "sources": sources_list,
                "route_taken": final_state.get("route_taken", "unknown"),
                "meta": final_state.get("meta", {}),
            }
        except Exception as exc:
            logger.warning("Remote graph run failed: %s. Falling back to local execution.", exc)

    # Local fallback execution
    invoke_config: Dict[str, Any] = {}
    if is_tracing_enabled():
        invoke_config["run_name"] = "teknofest-rag-query"
        invoke_config["metadata"] = {"question_preview": question[:120]}

    if metadata:
        invoke_config.setdefault("metadata", {}).update(metadata)

    if callbacks:
        invoke_config["callbacks"] = callbacks

    invoke_kwargs: Dict[str, Any] = {}
    if invoke_config:
        invoke_kwargs["config"] = invoke_config

    final_state = await graph.ainvoke(initial_state, **invoke_kwargs)
    sources_list = _build_sources_list(final_state.get("context_chunks", []))

    return {
        "answer": final_state.get("answer", ""),
        "sources": sources_list,
        "route_taken": final_state.get("route_taken", "unknown"),
        "meta": final_state.get("meta", {}),
    }


async def run_graph_stream(
    graph,
    question: str,
    chat_history: List[Dict[str, str]] = None,
    callbacks: list | None = None,
    metadata: Dict[str, Any] | None = None,
):
    """Asynchronously execute graph and yield tokens/metadata. Runs remotely if dev server is active."""
    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or [],
        "rephrased_question": "",
    }

    # Try streaming from remote langgraph dev server
    remote_url = await _get_remote_client_url()
    if remote_url:
        try:
            client = get_client(url=remote_url)
            thread_id = (metadata or {}).get("session_id") or str(uuid.uuid4())
            
            try:
                await client.threads.get(thread_id)
            except Exception:
                await client.threads.create(thread_id=thread_id)
                
            final_state = None
            async for part in client.runs.stream(
                thread_id=thread_id,
                assistant_id="teknofest_rag",
                input=initial_state,
                stream_mode="events"
            ):
                if part.event == "events":
                    event_data = part.data
                    event_name = event_data.get("event")
                    name = event_data.get("name")
                    tags = event_data.get("tags") or []
                    
                    if event_name == "on_chat_model_stream" and "final_synthesizer" in tags:
                        chunk = event_data.get("data", {}).get("chunk", {})
                        content = chunk.get("content") if isinstance(chunk, dict) else getattr(chunk, "content", "")
                        if content:
                            yield {"type": "token", "content": content}
                    elif event_name == "on_chain_end" and name == "teknofest_rag":
                        final_state = event_data.get("data", {}).get("output", {})

            if final_state:
                sources_list = _build_sources_list(final_state.get("context_chunks", []))
                yield {
                    "type": "done",
                    "sources": sources_list,
                    "route_taken": final_state.get("route_taken", "unknown"),
                    "meta": final_state.get("meta", {}),
                }
                return
        except Exception as exc:
            logger.warning("Remote graph stream failed: %s. Falling back to local execution.", exc)

    # Local fallback streaming
    invoke_config: Dict[str, Any] = {}
    if is_tracing_enabled():
        invoke_config["run_name"] = "teknofest-rag-query"
        invoke_config["metadata"] = {"question_preview": question[:120]}

    if metadata:
        invoke_config.setdefault("metadata", {}).update(metadata)

    if callbacks:
        invoke_config["callbacks"] = callbacks

    invoke_kwargs: Dict[str, Any] = {}
    if invoke_config:
        invoke_kwargs["config"] = invoke_config

    final_state = None
    async for event in graph.astream_events(initial_state, version="v2", **invoke_kwargs):
        kind = event["event"]
        tags = event.get("tags", [])
        
        if kind == "on_chat_model_stream" and "final_synthesizer" in tags:
            content = event["data"]["chunk"].content
            if content:
                yield {"type": "token", "content": content}
                
        elif kind == "on_chain_end":
            output = event["data"].get("output")
            if isinstance(output, dict) and "answer" in output and "context_chunks" in output:
                final_state = output

    sources_list = []
    route_taken = "unknown"
    if final_state:
        route_taken = final_state.get("route_taken", "unknown")
        sources_list = _build_sources_list(final_state.get("context_chunks", []))

    yield {
        "type": "done",
        "sources": sources_list,
        "route_taken": route_taken,
        "meta": final_state.get("meta", {}) if final_state else {},
    }



# ---------------------------------------------------------------------------
# PNG export helper (unchanged)
# ---------------------------------------------------------------------------


def export_graph_png(settings: Settings, output_path: str | Path) -> Path:
    """Export the LangGraph pipeline as a PNG diagram."""
    graph = build_teknofest_graph(settings=settings)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.get_graph().draw_png(str(path))
    return path


# ---------------------------------------------------------------------------
# LangGraph Studio / External Tools Export
# ---------------------------------------------------------------------------

from app.config import get_settings as _get_settings

# Create a default compiled graph instance for LangGraph Studio
_default_settings = _get_settings()
graph = build_teknofest_graph(settings=_default_settings)

if __name__ == "__main__":
    out = export_graph_png(_default_settings, Path(__file__).resolve().parent / "langgraph_teknofest.png")
    print(f"LangGraph PNG yazıldı: {out}")
