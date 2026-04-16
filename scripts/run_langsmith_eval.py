"""
run_langsmith_eval.py
=====================
Push the local eval dataset to LangSmith and run a batch evaluation.

What it does
------------
1. Creates (or reuses) a LangSmith dataset named after ``LANGCHAIN_PROJECT``.
2. Uploads all Q&A pairs from ``data/eval_dataset.json`` as dataset examples.
3. Runs the full RAG pipeline over each example and records results as a
   LangSmith experiment run.
4. Prints a Recall@K summary and the LangSmith experiment URL.

Usage
-----
::

    # Ensure your .env has LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY set
    python scripts/run_langsmith_eval.py

    # Custom k
    python scripts/run_langsmith_eval.py --k 5

Prerequisites
-------------
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_API_KEY=<your key>
- LANGCHAIN_PROJECT=teknofest-rag  (or any name)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.tracing import get_langsmith_client, init_langsmith, is_tracing_enabled
from app.rag.evaluator import load_eval_dataset, recall_at_k, tag_failures
from app.rag.retrievers import (
    build_local_docs_retriever,
    build_teknofest_site_retriever,
    retrieve_from_vectorstore,
)
from app.rag.embedding_service import get_embedding_service


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------


def ensure_dataset(client, dataset_name: str, examples: List[Dict[str, Any]]):
    """
    Create (or reuse) a LangSmith dataset and upload examples.

    Existing examples with the same input are skipped to avoid duplication.
    """
    # Get or create dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Reusing existing LangSmith dataset: '{dataset_name}' (id={dataset.id})")
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="TEKNOFEST RAG ground-truth retrieval evaluation dataset",
        )
        print(f"Created new LangSmith dataset: '{dataset_name}' (id={dataset.id})")

    # Fetch existing inputs to avoid duplicates
    existing = {
        ex.inputs.get("query", "")
        for ex in client.list_examples(dataset_id=dataset.id)
    }

    new_examples = [e for e in examples if e["query"] not in existing]
    if new_examples:
        client.create_examples(
            inputs=[{"query": e["query"]} for e in new_examples],
            outputs=[{"expected_keywords": e.get("expected_keywords", [])} for e in new_examples],
            dataset_id=dataset.id,
        )
        print(f"Uploaded {len(new_examples)} new examples ({len(existing)} already existed).")
    else:
        print("All examples already exist in the dataset — nothing to upload.")

    return dataset


# ---------------------------------------------------------------------------
# Evaluation target function
# ---------------------------------------------------------------------------


def make_retrieval_evaluator(settings, k: int):
    """Returns a function that retrieves top-K chunks for a given query."""
    # Build retrievers once (they hold Chroma connections)
    local_vs = build_local_docs_retriever(settings)
    site_vs = build_teknofest_site_retriever(settings)

    def retrieve_for_eval(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        local_chunks = retrieve_from_vectorstore(
            local_vs, query, source_type="local_docs", k=k
        )
        site_chunks = retrieve_from_vectorstore(
            site_vs, query, source_type="teknofest_site", k=k
        )
        all_chunks = sorted(
            local_chunks + site_chunks, key=lambda c: c.score or 0.0
        )
        top_k = all_chunks[:k]
        return {
            "top_k_contents": [c.content[:300] for c in top_k],
            "top_k_sources": [c.source for c in top_k],
            "top_k_scores": [c.score for c in top_k],
            "retrieved_count": len(top_k),
        }

    return retrieve_for_eval


# ---------------------------------------------------------------------------
# LangSmith evaluator functions
# ---------------------------------------------------------------------------


def recall_evaluator(run, example):
    """LangSmith evaluator: Recall@K using expected_keywords."""
    from langsmith.schemas import EvaluationResult

    expected_keywords = (example.outputs or {}).get("expected_keywords", [])
    retrieved_contents = (run.outputs or {}).get("top_k_contents", [])
    combined = " ".join(retrieved_contents).lower()

    if not expected_keywords:
        score = 1.0
    else:
        hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
        score = hits / len(expected_keywords)

    return EvaluationResult(
        key="recall_at_k",
        score=score,
        comment=f"Found {int(score * len(expected_keywords))}/{len(expected_keywords)} keywords",
    )


def result_count_evaluator(run, example):
    """LangSmith evaluator: did we get any results at all?"""
    from langsmith.schemas import EvaluationResult

    count = (run.outputs or {}).get("retrieved_count", 0)
    return EvaluationResult(
        key="has_results",
        score=1.0 if count > 0 else 0.0,
        comment=f"Retrieved {count} chunks",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LangSmith RAG Evaluation Runner")
    settings = get_settings()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.eval_dataset_path,
        help="Path to eval_dataset.json",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=settings.retrieval_final_k,
        help="Top-K for retrieval during evaluation",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="retrieval-eval",
        help="Prefix for the LangSmith experiment name",
    )
    args = parser.parse_args()

    # ---- Tracing check ----
    init_langsmith(settings)
    if not is_tracing_enabled():
        print(
            "\n[ERROR] LangSmith tracing is not enabled.\n"
            "Set the following in your .env file and restart:\n"
            "  LANGCHAIN_TRACING_V2=true\n"
            "  LANGCHAIN_API_KEY=<your LangSmith API key>\n"
            "  LANGCHAIN_PROJECT=teknofest-rag\n"
        )
        sys.exit(1)

    client = get_langsmith_client()
    if client is None:
        print("[ERROR] Could not build LangSmith client.")
        sys.exit(1)

    if not args.dataset.exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        sys.exit(1)

    # ---- Load dataset ----
    examples = load_eval_dataset(args.dataset)
    print(f"\nLoaded {len(examples)} test cases from {args.dataset}")
    print(f"LangSmith project: {settings.langsmith_project}")
    print(f"Retrieval k: {args.k}\n")

    # ---- Warm up embedding service (validates API key) ----
    get_embedding_service(settings)

    # ---- Push to LangSmith ----
    dataset_name = f"{settings.langsmith_project}-eval"
    dataset = ensure_dataset(client, dataset_name, examples)

    # ---- Run evaluation ----
    print(f"\nRunning LangSmith evaluation experiment (prefix='{args.experiment_prefix}')…")
    try:
        from langsmith.evaluation import evaluate

        results = evaluate(
            make_retrieval_evaluator(settings, k=args.k),
            data=dataset_name,
            evaluators=[recall_evaluator, result_count_evaluator],
            experiment_prefix=args.experiment_prefix,
            metadata={
                "k": args.k,
                "embedding_model": settings.embedding_model_name,
                "project": settings.langsmith_project,
            },
        )

        print("\n" + "=" * 60)
        print(f"  Experiment complete!")
        print(f"  Results URL: {results.experiment_results_url}")
        print("=" * 60 + "\n")

    except Exception as exc:
        print(f"[ERROR] LangSmith evaluation failed: {exc}")
        print("Falling back to local Recall@K report…\n")

        # Graceful fallback: run locally and print results
        _run_local_fallback(settings, examples, k=args.k)


def _run_local_fallback(settings, examples, k: int) -> None:
    """Print a local Recall@K report when LangSmith evaluation fails."""
    local_vs = build_local_docs_retriever(settings)
    site_vs = build_teknofest_site_retriever(settings)
    recalls = []

    for i, case in enumerate(examples, start=1):
        query = case["query"]
        keywords = case.get("expected_keywords", [])

        local_chunks = retrieve_from_vectorstore(
            local_vs, query, source_type="local_docs", k=k
        )
        site_chunks = retrieve_from_vectorstore(
            site_vs, query, source_type="teknofest_site", k=k
        )
        top_k = sorted(local_chunks + site_chunks, key=lambda c: c.score or 0.0)[:k]

        r = recall_at_k(top_k, keywords, k=k)
        recalls.append(r)
        status = "✓" if r >= 0.5 else "✗"
        print(f"[{i:02d}] {status}  Recall@{k}={r:.2f}  | {query[:60]}")

    avg = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"\nAverage Recall@{k}: {avg:.3f}  ({len(examples)} queries)")


if __name__ == "__main__":
    main()
