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
from app.rag.prompts import INTENT_CLASSIFICATION_PROMPT, SYSTEM_PROMPT_BASE
from app.rag.reranker import rerank_chunks
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

RouteLiteral = Literal["direct", "local", "site", "tavily"]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, str]]
    intent: Literal["TEKNOFEST", "DIGER"]
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


def _build_llm(settings: Settings, temperature: float = 0.2):
    return get_llm_service(settings).get_chat_model(temperature=temperature)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def node_intent_classification(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.0)
    prompt = INTENT_CLASSIFICATION_PROMPT.format(question=state["question"])
    res = await llm.ainvoke([{"role": "user", "content": prompt}])
    label_raw = (res.content or "").strip().upper()
    intent: Literal["TEKNOFEST", "DIGER"] = (
        "TEKNOFEST" if "TEKNOFEST" in label_raw else "DIGER"
    )
    state["intent"] = intent
    state.setdefault("meta", {})["intent"] = intent
    return state


def _decide_next_after_intent(state: GraphState) -> str:
    return "local_rag" if state.get("intent") == "TEKNOFEST" else "direct_llm"


# ---- Local RAG ----

def node_local_rag(state: GraphState, settings: Settings) -> GraphState:
    vs = build_local_docs_retriever(settings)
    chunks = retrieve_from_vectorstore(
        vs, query=state["question"],
        source_type="local_docs",
        k=settings.retrieval_top_k,
    )
    state.setdefault("retrieved_chunks", []).extend(chunks)
    state.setdefault("meta", {})["local_confidence"] = estimate_chunk_confidence(chunks)
    if chunks and estimate_chunk_confidence(chunks) >= settings.rag_confidence_threshold:
        return state
    # Not enough confidence — mark for fallback
    state["meta"]["local_rag_empty"] = True
    # Remove local chunks so cascade continues
    state["retrieved_chunks"] = [
        c for c in state.get("retrieved_chunks", []) if c.source_type != "local_docs"
    ]
    return state


def _decide_after_local_rag(state: GraphState) -> str:
    has_local = any(
        c.source_type == "local_docs" for c in state.get("retrieved_chunks", [])
    )
    
    question = state.get("question", "").lower()
    keywords = ["bu yıl", "bu sene", "2026", "güncel", "yeni", "this year", "هذه السنة", "هذا العام", "الآن", "حاليا", "current"]
    needs_current = any(kw in question for kw in keywords)
    
    if has_local and not needs_current:
        return "reranker"
    return "teknofest_web"


# ---- Teknofest site RAG ----

def node_teknofest_web(state: GraphState, settings: Settings) -> GraphState:
    vs = build_teknofest_site_retriever(settings)
    chunks = retrieve_from_vectorstore(
        vs, query=state["question"],
        source_type="teknofest_site",
        k=settings.retrieval_top_k,
    )
    state.setdefault("retrieved_chunks", []).extend(chunks)
    state.setdefault("meta", {})["site_confidence"] = estimate_chunk_confidence(chunks)
    if chunks and estimate_chunk_confidence(chunks) >= settings.rag_confidence_threshold:
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
    return "reranker" if has_site else "tavily_web"


# ---- Tavily web fallback ----

def node_tavily_web(state: GraphState, settings: Settings) -> GraphState:
    try:
        tavily_tool = build_tavily_tool(settings)
        chunks = retrieve_from_tavily(tavily_tool, query=state["question"])
        state.setdefault("retrieved_chunks", []).extend(chunks)
    except RuntimeError as exc:
        logger.warning("Tavily unavailable: %s", exc)
    return state


# ---- Reranker ----

async def node_reranker(state: GraphState, settings: Settings) -> GraphState:
    retrieved = state.get("retrieved_chunks", [])
    if not retrieved:
        state["context_chunks"] = []
        return state

    if settings.reranker_enabled:
        llm = _build_llm(settings, temperature=0.0)
        final = await rerank_chunks(
            query=state["question"],
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
    llm = _build_llm(settings, temperature=0.2)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
    for msg in state.get("chat_history", []):
        messages.append(msg)
    messages.append({"role": "user", "content": state["question"]})
    res = await llm.ainvoke(messages)
    state["answer"] = res.content or ""
    state["route_taken"] = "direct"
    state["context_chunks"] = []
    state["retrieved_chunks"] = []
    return state


# ---- Answer synthesizer ----

async def node_answer_synthesizer(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.1)

    context_block = state.get("context_str") or ""
    if not context_block:
        # Fallback: build on the fly
        context_block, _ = build_context(state.get("context_chunks", []))

    user_content = (
        f"Soru:\n{state['question']}\n\n"
        "Aşağıda ilgili olabilecek bağlam parçaları verilmiştir. "
        "Bu bağlamları kullanarak, mümkün olduğunca TEKNOFEST odaklı, "
        "kaynaklara dayalı bir cevap üret. Cevabın sonunda, hangi kaynaklardan "
        "yararlandığını kısaca listele.\n\n"
        f"BAĞLAM:\n{context_block}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
    for msg in state.get("chat_history", []):
        messages.append(msg)
    messages.append({"role": "user", "content": user_content})

    # Store prompt preview for LangSmith trace metadata
    state.setdefault("meta", {})["prompt_preview"] = user_content[:500]

    res = await llm.ainvoke(
        messages,
        # run_name must go inside config=, NOT as a top-level kwarg to ainvoke().
        # Passing it top-level causes a TypeError that surfaces as HTTP 500.
        config={"run_name": "llm_generation"} if is_tracing_enabled() else None,
    )
    state["answer"] = res.content or ""

    # Determine route label from dominant source type
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

    if hal_status == "suspicious":
        state["answer"] = "Insufficient reliable information available."

    # --- Evaluation logging ---
    retrieved = state.get("retrieved_chunks", [])
    selected = state.get("context_chunks", [])
    _log_eval(settings, state, retrieved, selected, hal_status)

    return state


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

    async def answer_synthesizer_node(s: GraphState) -> GraphState:
        return await node_answer_synthesizer(s, settings)

    async def hallucination_guard_node(s: GraphState) -> GraphState:
        return await node_hallucination_guard(s, settings)

    # ---- Register nodes ----
    workflow.add_node("intent", intent_node)
    workflow.add_node("local_rag", local_rag_node)
    workflow.add_node("teknofest_web", teknofest_web_node)
    workflow.add_node("tavily_web", tavily_web_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("direct_llm", direct_llm_node)
    workflow.add_node("answer_synthesizer", answer_synthesizer_node)
    workflow.add_node("hallucination_guard", hallucination_guard_node)

    # ---- Edges ----
    workflow.set_entry_point("intent")

    workflow.add_conditional_edges("intent", _decide_next_after_intent, {
        "local_rag": "local_rag",
        "direct_llm": "direct_llm",
    })
    workflow.add_conditional_edges("local_rag", _decide_after_local_rag, {
        "reranker": "reranker",
        "teknofest_web": "teknofest_web",
    })
    workflow.add_conditional_edges("teknofest_web", _decide_after_teknofest_site, {
        "reranker": "reranker",
        "tavily_web": "tavily_web",
    })

    workflow.add_edge("tavily_web", "reranker")
    workflow.add_edge("reranker", "context_builder")
    workflow.add_edge("context_builder", "answer_synthesizer")
    workflow.add_edge("answer_synthesizer", "hallucination_guard")
    workflow.add_edge("direct_llm", "hallucination_guard")
    workflow.add_edge("hallucination_guard", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public run helper
# ---------------------------------------------------------------------------


async def run_graph(graph, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Wrapper used by FastAPI and tests."""
    initial_state: GraphState = {"question": question, "chat_history": chat_history or []}

    # When LangSmith tracing is active, pass a run_name so each query appears
    # as a named root trace in the LangSmith project.
    invoke_kwargs: Dict[str, Any] = {}
    if is_tracing_enabled():
        invoke_kwargs["config"] = {
            "run_name": "teknofest-rag-query",
            "metadata": {"question_preview": question[:120]},
        }

    final_state = await graph.ainvoke(initial_state, **invoke_kwargs)
    return {
        "answer": final_state.get("answer", ""),
        "sources": [
            {
                "type": ch.source_type,
                "metadata": ch.metadata,
                "score": ch.score,
            }
            for ch in final_state.get("context_chunks", [])
        ],
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
