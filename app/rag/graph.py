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

logger = logging.getLogger(__name__)

RouteLiteral = Literal["direct", "local", "site", "tavily", "llm_knowledge", "kisisel"]

PERSONALITY_PREFIX = """Sen Türkiye'nin en büyük teknoloji festivali TEKNOFEST'in yardımcı chatbot'usun.
Kullanıcılarla samimi, sıcak ve destekleyici bir dille konuşursun.
Sanki yarışmacıların en büyük destekçisiymiş gibi hissettirirsin.
Cevaplarını her zaman Türkçe verirsin.
Özellikle kategori veya takım sorularında yarışmacıların heyecanını paylaş."""



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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def node_intent_classification(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.0, purpose="tavily")
    from datetime import datetime
    current_date = datetime.now().strftime("%d %B %Y, %A")
    prompt = f"Bugünün tarihi: {current_date}\n\n" + INTENT_CLASSIFICATION_PROMPT.format(question=state["question"])
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
        return "rephrase"
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
    except Exception as exc:
        logger.warning("Rephrase LLM call failed: %s, using original question", exc)
        state["rephrased_question"] = state["question"]
        state.setdefault("meta", {})["rephrase_used"] = False
        state.setdefault("meta", {})["rephrase_error"] = str(exc)
    
    return state


# ---- Local RAG ----

def node_local_rag(state: GraphState, settings: Settings) -> GraphState:
    vs = build_local_docs_retriever(settings)
    query = normalize_turkish_query(state.get("rephrased_question") or state["question"])
    chunks = retrieve_from_vectorstore(
        vs, query=query,
        source_type="local_docs",
        k=settings.retrieval_top_k,
    )
    
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
    keywords = ["bu yıl", "bu sene", "2026", "güncel", "yeni", "this year", "هذه السنة", "هذا العام", "الآن", "حاليا", "current"]
    needs_current = any(kw in question for kw in keywords)

    if has_local and not needs_current:
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
    if has_site:
        return "reranker"
    
    question = (state.get("rephrased_question") or state.get("question", "")).lower()
    keywords = ["bu yıl", "bu sene", "2026", "güncel", "yeni", "this year", "هذه السنة", "هذا العام", "الآن", "حاليا", "current"]
    needs_current = any(kw in question for kw in keywords)
    
    if needs_current:
        return "tavily_web"
    else:
        return "llm_knowledge"


# ---- Tavily web fallback ----

def node_tavily_web(state: GraphState, settings: Settings) -> GraphState:
    try:
        tavily_tool = build_tavily_tool(settings)
        query = normalize_turkish_query(state.get("rephrased_question") or state["question"])
        chunks = retrieve_from_tavily(tavily_tool, query=query)
        state.setdefault("retrieved_chunks", []).extend(chunks)
    except RuntimeError as exc:
        logger.warning("Tavily unavailable: %s", exc)
    return state


def _decide_after_tavily(state: GraphState) -> str:
    """If Tavily produced chunks, proceed to reranker; otherwise fall back to LLM knowledge."""
    has_tavily = any(
        c.source_type == "tavily" for c in state.get("retrieved_chunks", [])
    )
    # Also check if we still have any retrieved chunks at all from earlier stages
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
        max_total_chars=6_000,
        min_score=None,   # score filtering already done via confidence threshold
        min_rerank=None,  # rerank threshold left to the reranker node
    )
    state["context_chunks"] = selected
    state["context_str"] = context_str
    state.setdefault("meta", {})["context_chunks_count"] = len(selected)
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
    sys_prompt = PERSONALITY_PREFIX + "\n\nSen TEKNOFEST'in sıcak ve samimi chatbot'usun. Sana kişisel bir soru soruldu. Eğlenceli ve destekleyici bir şekilde cevap ver, sonra konuyu nazikçe TEKNOFEST'e çek. Örnek: 'En sevdiğim kategori mi? Senin kategorin tabii ki! Hangi yarışmaya hazırlanıyorsun?'"
    ans_res = await llm_gen.ainvoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": state["question"]}
    ])
    state["answer"] = ans_res.content
    state["route_taken"] = "kisisel"
    state["context_chunks"] = []
    state["retrieved_chunks"] = []
    return state


# ---- LLM Knowledge Fallback ----

async def node_llm_knowledge(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.3, purpose="main")
    sys_prompt = PERSONALITY_PREFIX + "\n\nSen bir TEKNOFEST uzmanısın. Sana sağlanan belgelerden bu soruya cevap bulunamadı. TEKNOFEST hakkındaki genel bilginle soruyu yanıtla. Emin olmadığın bilgileri 'resmi kaynakları kontrol etmenizi öneririm' diyerek belirt. Asla TEKNOFEST dışı konularda bilgi üretme."
    
    res = await llm.ainvoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": state.get("rephrased_question") or state["question"]}
    ])
    
    state["answer"] = res.content
    state["route_taken"] = "llm_knowledge"
    state["context_chunks"] = []
    state["retrieved_chunks"] = []
    return state


# ---- Answer synthesizer (F.3 — Agent entegrasyonu) ----

async def node_answer_synthesizer(state: GraphState, settings: Settings) -> GraphState:
    """
    F.3 — RAG bağlamını agent'e geçirir; agent tool calling loop ile
    gerekirse gerçek zamanlı bilgi çekerek cevabı üretir.
    route_taken ve prompt_preview mantığı korunur.
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
    result = await hallucination_check(
        settings=settings,
        question=state["question"],
        answer=state.get("answer", ""),
        context_chunks=state.get("context_chunks", []),
    )
    state.setdefault("meta", {})["hallucination_check"] = result
    hal_status = result.get("status", "unknown")

    # Routes that generate answers without RAG context — never override them
    bypass_routes = {"llm_knowledge", "kisisel", "direct"}
    current_route = state.get("route_taken", "")

    if current_route not in bypass_routes and not state.get("context_chunks"):
        # ONLY force rejection if there's truly NO context
        # If we have context, trust the agent's answer even if partially relevant
        state["answer"] = "Sağlanan belgelerden ilgili bilgi bulamadım."

    # --- Evaluation logging ---
    retrieved = state.get("retrieved_chunks", [])
    selected = state.get("context_chunks", [])
    _log_eval(settings, state, retrieved, selected, hal_status)

    return state


def _decide_after_hallucination_guard(state: GraphState) -> str:
    """
    If agent rejected answer ('bulunamadı') but we have context,
    route to llm_knowledge for general knowledge synthesis.
    """
    answer = state.get("answer", "").strip()
    has_context = bool(state.get("context_chunks"))
    is_rejection = "bulunamadı" in answer.lower() or answer == "Sağlanan belgelerden ilgili bilgi bulamadım."
    current_route = state.get("route_taken", "")
    
    # If we have context but agent said "not found", fallback to general knowledge
    if has_context and is_rejection and current_route in ("local", "site"):
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

    # ---- Register nodes ----
    workflow.add_node("intent", intent_node)
    workflow.add_node("rephrase", rephrase_node)                   # D.3
    workflow.add_node("local_rag", local_rag_node)
    workflow.add_node("teknofest_web", teknofest_web_node)
    workflow.add_node("tavily_web", tavily_web_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("direct_llm", direct_llm_node)
    workflow.add_node("kisisel_llm", kisisel_llm_node)
    workflow.add_node("llm_knowledge", llm_knowledge_node)
    workflow.add_node("answer_synthesizer", answer_synthesizer_node)
    workflow.add_node("hallucination_guard", hallucination_guard_node)

    # ---- Edges ----
    # intent → conditional (rephrase veya direct_llm)
    workflow.set_entry_point("intent")
    workflow.add_conditional_edges("intent", _decide_next_after_intent, {
        "rephrase": "rephrase",
        "kisisel_llm": "kisisel_llm",
        "direct_llm": "direct_llm",
    })
    workflow.add_edge("rephrase", "local_rag")
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
    workflow.add_edge("context_builder", "answer_synthesizer")
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
# Public run helper
# ---------------------------------------------------------------------------


async def run_graph(
    graph,
    question: str,
    chat_history: List[Dict[str, str]] = None,
    callbacks: list | None = None,          # E.3 — Langfuse / LangSmith callbacks
    metadata: Dict[str, Any] | None = None, # Sprint 3 — session/user metadata
) -> Dict[str, Any]:
    """Wrapper used by FastAPI and tests."""
    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or [],
        "rephrased_question": "",   # D.2 — node_rephrase tarafından doldurulur
    }

    # Invoke config — LangSmith run_name + E.3 Langfuse callbacks
    invoke_config: Dict[str, Any] = {}
    if is_tracing_enabled():
        invoke_config["run_name"] = "teknofest-rag-query"
        invoke_config["metadata"] = {"question_preview": question[:120]}

    # Sprint 3 — session/user metadata merge
    if metadata:
        invoke_config.setdefault("metadata", {}).update(metadata)

    # LangSmith ve Langfuse aynı callbacks listesinde birlikte çalışabilir
    if callbacks:
        invoke_config["callbacks"] = callbacks

    invoke_kwargs: Dict[str, Any] = {}
    if invoke_config:
        invoke_kwargs["config"] = invoke_config

    final_state = await graph.ainvoke(initial_state, **invoke_kwargs)
    # Kaynak listesini güvenilirlik etiketleriyle oluştur
    sources_list = []
    seen_sources = set()  # deduplicate by source path/url
    for ch in final_state.get("context_chunks", []):
        # Unique source identifier
        src_id = (
            ch.metadata.get("source")
            or ch.metadata.get("url")
            or ch.metadata.get("crawl_source")
            or ""
        )
        if src_id and src_id in seen_sources:
            continue
        if src_id:
            seen_sources.add(src_id)

        source_entry = {
            "type": ch.source_type,
            "metadata": ch.metadata,
            "score": round(ch.score, 3) if ch.score is not None else None,
            "content_preview": ch.content[:200] if hasattr(ch, "content") else "",
        }
        
        # Güvenilirlik etiketleri
        if ch.source_type == "local_docs":
            source_entry["trust_level"] = "high"
            source_entry["trust_label"] = "Resmi Doküman (Yerel)"
        elif ch.source_type == "teknofest_site":
            source_entry["trust_level"] = "high"
            source_entry["trust_label"] = "TEKNOFEST Resmi Sitesi"
        elif ch.source_type == "tavily":
            url = ch.metadata.get("url", "")
            if any(d in url for d in ["teknofest.org", "cdn.teknofest.org", ".gov.tr"]):
                source_entry["trust_level"] = "high"
                source_entry["trust_label"] = "Resmi Kaynak"
            else:
                source_entry["trust_level"] = "medium"
                source_entry["trust_label"] = "Web Kaynağı"
                
        sources_list.append(source_entry)

    return {
        "answer": final_state.get("answer", ""),
        "sources": sources_list,
        "route_taken": final_state.get("route_taken", "unknown"),
        "meta": final_state.get("meta", {}),
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


if __name__ == "__main__":
    from app.config import get_settings as _get_settings

    _settings = _get_settings()
    out = export_graph_png(_settings, Path(__file__).resolve().parent / "langgraph_teknofest.png")
    print(f"LangGraph PNG yazıldı: {out}")
