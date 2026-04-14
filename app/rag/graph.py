from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.llm import get_llm_service
from .hallucination_guard import hallucination_check
from .prompts import INTENT_CLASSIFICATION_PROMPT, SYSTEM_PROMPT_BASE
from .retrievers import (
    RetrievedChunk,
    build_local_docs_retriever,
    build_teknofest_site_retriever,
    build_tavily_tool,
    estimate_chunk_confidence,
    retrieve_from_tavily,
    retrieve_from_vectorstore,
)


RouteLiteral = Literal["direct", "local", "site", "tavily"]


class GraphState(TypedDict, total=False):
    question: str
    intent: Literal["TEKNOFEST", "DIGER"]
    context_chunks: List[RetrievedChunk]
    answer: str
    route_taken: RouteLiteral
    meta: Dict[str, Any]


def _build_llm(settings: Settings, temperature: float = 0.2):
    return get_llm_service(settings).get_chat_model(temperature=temperature)


async def node_intent_classification(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.0)
    prompt = INTENT_CLASSIFICATION_PROMPT.format(question=state["question"])
    res = await llm.ainvoke([{"role": "user", "content": prompt}])
    label_raw = (res.content or "").strip().upper()
    intent: Literal["TEKNOFEST", "DIGER"]
    if "TEKNOFEST" in label_raw:
        intent = "TEKNOFEST"
    else:
        intent = "DIGER"
    state["intent"] = intent
    return state


def _decide_next_after_intent(state: GraphState) -> str:
    if state.get("intent") == "TEKNOFEST":
        return "local_rag"
    return "direct_llm"


def node_local_rag(state: GraphState, settings: Settings) -> GraphState:
    vs = build_local_docs_retriever(settings)
    chunks = retrieve_from_vectorstore(vs, query=state["question"], source_type="local_docs", k=5)
    state.setdefault("context_chunks", [])
    state["context_chunks"].extend(chunks)

    confidence = estimate_chunk_confidence(chunks)
    state.setdefault("meta", {})
    state["meta"]["local_confidence"] = confidence
    if chunks and confidence >= settings.rag_confidence_threshold:
        return state
    state["meta"]["local_rag_empty"] = True
    state["context_chunks"] = []
    return state


def _decide_after_local_rag(state: GraphState) -> str:
    if state.get("context_chunks"):
        return "answer_synthesizer"
    return "teknofest_web"


def node_teknofest_web(state: GraphState, settings: Settings) -> GraphState:
    vs = build_teknofest_site_retriever(settings)
    chunks = retrieve_from_vectorstore(vs, query=state["question"], source_type="teknofest_site", k=5)
    state.setdefault("context_chunks", [])
    state["context_chunks"].extend(chunks)
    confidence = estimate_chunk_confidence(chunks)
    state.setdefault("meta", {})
    state["meta"]["site_confidence"] = confidence
    if chunks and confidence >= settings.rag_confidence_threshold:
        return state
    state["meta"]["teknofest_site_empty"] = True
    state["context_chunks"] = []
    return state


def _decide_after_teknofest_site(state: GraphState) -> str:
    if state.get("context_chunks"):
        return "answer_synthesizer"
    return "tavily_web"


def node_tavily_web(state: GraphState, settings: Settings) -> GraphState:
    tavily_tool = build_tavily_tool(settings)
    chunks = retrieve_from_tavily(tavily_tool, query=state["question"])
    state.setdefault("context_chunks", [])
    state["context_chunks"].extend(chunks)
    return state


async def node_direct_llm(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": state["question"]},
    ]
    res = await llm.ainvoke(messages)
    state["answer"] = res.content or ""
    state["route_taken"] = "direct"
    return state


async def node_answer_synthesizer(state: GraphState, settings: Settings) -> GraphState:
    llm = _build_llm(settings, temperature=0.1)

    context_texts = []
    for idx, ch in enumerate(state.get("context_chunks", []), start=1):
        src = ch.metadata.get("source") or ch.metadata.get("file_path") or ch.metadata.get("url") or "bilinmeyen kaynak"
        context_texts.append(f"[{idx}] ({ch.source_type}) {src}\n{ch.content}")
    context_block = "\n\n".join(context_texts)

    user_content = (
        f"Soru:\n{state['question']}\n\n"
        "Aşağıda ilgili olabilecek bağlam parçaları verilmiştir. "
        "Bu bağlamları kullanarak, mümkün olduğunca TEKNOFEST odaklı, "
        "kaynaklara dayalı bir cevap üret. Cevabın sonunda, hangi kaynaklardan "
        "yararlandığını kısaca listele.\n\n"
        f"BAĞLAM:\n{context_block}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": user_content},
    ]
    res = await llm.ainvoke(messages)
    state["answer"] = res.content or ""

    # route label, context içindeki baskın source_type'a göre
    types = {ch.source_type for ch in state.get("context_chunks", [])}
    route: RouteLiteral = "local"
    if "teknofest_site" in types:
        route = "site"
    if "tavily" in types and len(types) == 1:
        route = "tavily"
    state["route_taken"] = route
    return state


async def node_hallucination_guard(state: GraphState, settings: Settings) -> GraphState:
    result = await hallucination_check(
        settings=settings,
        question=state["question"],
        answer=state.get("answer", ""),
        context_chunks=state.get("context_chunks", []),
    )
    state.setdefault("meta", {})
    state["meta"]["hallucination_check"] = result
    if result.get("status") == "suspicious":
        state["answer"] = "Insufficient reliable information available."
    return state


def build_teknofest_graph(settings: Settings):
    """
    LangGraph tablo tanımı.
    """
    workflow = StateGraph(GraphState)

    async def intent_node(state: GraphState) -> GraphState:
        return await node_intent_classification(state, settings)

    def local_rag_node(state: GraphState) -> GraphState:
        return node_local_rag(state, settings)

    def teknofest_web_node(state: GraphState) -> GraphState:
        return node_teknofest_web(state, settings)

    def tavily_web_node(state: GraphState) -> GraphState:
        return node_tavily_web(state, settings)

    async def direct_llm_node(state: GraphState) -> GraphState:
        return await node_direct_llm(state, settings)

    async def answer_synthesizer_node(state: GraphState) -> GraphState:
        return await node_answer_synthesizer(state, settings)

    async def hallucination_guard_node(state: GraphState) -> GraphState:
        return await node_hallucination_guard(state, settings)

    # Nodes
    workflow.add_node("intent", intent_node)
    workflow.add_node("local_rag", local_rag_node)
    workflow.add_node("teknofest_web", teknofest_web_node)
    workflow.add_node("tavily_web", tavily_web_node)
    workflow.add_node("direct_llm", direct_llm_node)
    workflow.add_node("answer_synthesizer", answer_synthesizer_node)
    workflow.add_node("hallucination_guard", hallucination_guard_node)

    # Edges
    workflow.set_entry_point("intent")
    workflow.add_conditional_edges("intent", _decide_next_after_intent, {
        "local_rag": "local_rag",
        "direct_llm": "direct_llm",
    })

    workflow.add_conditional_edges("local_rag", _decide_after_local_rag, {
        "answer_synthesizer": "answer_synthesizer",
        "teknofest_web": "teknofest_web",
    })

    workflow.add_conditional_edges("teknofest_web", _decide_after_teknofest_site, {
        "answer_synthesizer": "answer_synthesizer",
        "tavily_web": "tavily_web",
    })

    workflow.add_edge("tavily_web", "answer_synthesizer")
    workflow.add_edge("answer_synthesizer", "hallucination_guard")
    workflow.add_edge("direct_llm", "hallucination_guard")
    workflow.add_edge("hallucination_guard", END)

    return workflow.compile()


async def run_graph(graph, question: str) -> Dict[str, Any]:
    """
    Dış dünyadan kullanım için basit wrapper.
    """
    initial_state: GraphState = {
        "question": question,
    }
    final_state = await graph.ainvoke(initial_state)
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


def export_graph_png(settings: Settings, output_path: str | Path) -> Path:
    """
    LangGraph yapısını verilen yola PNG olarak çizer.

    Örnek kullanım:

        from app.config import get_settings
        from app.rag.graph import export_graph_png

        settings = get_settings()
        export_graph_png(settings, "app/langgraph_teknofest.png")
    """
    graph = build_teknofest_graph(settings=settings)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.get_graph().draw_png(str(path))
    return path


if __name__ == "__main__":  # istege bagli dogrudan calistirma
    from app.config import get_settings as _get_settings

    _settings = _get_settings()
    out = export_graph_png(_settings, Path(__file__).resolve().parent / "langgraph_teknofest.png")
    print(f"LangGraph PNG yazıldı: {out}")


