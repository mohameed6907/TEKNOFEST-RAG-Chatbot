from __future__ import annotations

from typing import Any, Dict, List

from app.config import Settings
from app.llm import get_llm_service
from .prompts import HALLUCINATION_CHECK_PROMPT
from .retrievers import RetrievedChunk


def summarize_sources_for_check(chunks: List[RetrievedChunk]) -> str:
    summaries = []
    for idx, ch in enumerate(chunks, start=1):
        meta = ch.metadata or {}
        src = meta.get("source") or meta.get("file_path") or meta.get("url") or "bilinmeyen kaynak"
        summaries.append(f"[{idx}] ({ch.source_type}) {src}: {ch.content[:300]}")
    return "\n".join(summaries)


async def hallucination_check(
    settings: Settings,
    question: str,
    answer: str,
    context_chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """
    Basit bir self-check: LLM'e cevabın kaynaklarla uyumlu olup olmadığını sorar.
    """
    if not context_chunks:
        return {"status": "unknown", "reason": "no_context"}

    llm = get_llm_service(settings).get_chat_model(temperature=0.0)

    source_summaries = summarize_sources_for_check(context_chunks)
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        question=question,
        answer=answer,
        source_summaries=source_summaries,
    )

    msg = [{"role": "user", "content": prompt}]
    res = await llm.ainvoke(msg)
    label = (res.content or "").strip().upper()

    if "SUPHELI" in label:
        return {"status": "suspicious", "raw_label": label}
    if "GUVENLI" in label:
        return {"status": "safe", "raw_label": label}
    return {"status": "unknown", "raw_label": label}

