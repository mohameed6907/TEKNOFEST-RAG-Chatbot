"""
run_eval.py
===========
CLI evaluation runner for the RAG pipeline.

Loads the ground-truth dataset (data/eval_dataset.json), runs each query
through the retriever only (no LLM generation, fast), and reports:

  - Recall@K per query
  - Average recall across the dataset
  - per-query failure tag summary

Usage
-----
::

    python scripts/run_eval.py
    python scripts/run_eval.py --dataset data/eval_dataset.json --k 5

Results are printed to stdout and written to RAG/eval_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.embedding_service import get_embedding_service
from app.rag.evaluator import load_eval_dataset, recall_at_k, tag_failures
from app.rag.retrievers import (
    build_local_docs_retriever,
    build_teknofest_site_retriever,
    retrieve_from_vectorstore,
)


def run_eval(dataset_path: Path, k: int) -> None:
    settings = get_settings()
    _ = get_embedding_service(settings)  # warm up

    print(f"\n{'='*60}")
    print(f"  TEKNOFEST RAG — Retrieval Evaluation (Recall@{k})")
    print(f"  Dataset: {dataset_path}")
    print(f"{'='*60}\n")

    dataset = load_eval_dataset(dataset_path)
    print(f"Loaded {len(dataset)} test cases.\n")

    # Build retrievers once
    local_vs = build_local_docs_retriever(settings)
    site_vs = build_teknofest_site_retriever(settings)

    results = []
    recalls = []

    for i, case in enumerate(dataset, start=1):
        query: str = case["query"]
        expected_keywords: list = case.get("expected_keywords", [])

        # Retrieve from both collections
        local_chunks = retrieve_from_vectorstore(
            local_vs, query, source_type="local_docs", k=k
        )
        site_chunks = retrieve_from_vectorstore(
            site_vs, query, source_type="teknofest_site", k=k
        )
        all_chunks = local_chunks + site_chunks
        # Sort by score (ascending = closer)
        all_chunks.sort(key=lambda c: c.score or 0.0)
        top_k = all_chunks[:k]

        r_at_k = recall_at_k(top_k, expected_keywords, k=k)
        recalls.append(r_at_k)

        tags = tag_failures(
            retrieved_chunks=top_k,
            answer="",
            hallucination_status="unknown",
            recall=r_at_k,
            recall_threshold=0.5,
        )

        status = "✓" if r_at_k >= 0.5 else "✗"
        print(
            f"[{i:02d}] {status}  Recall@{k}={r_at_k:.2f}  "
            f"tags={tags or '—'}  | {query[:60]}"
        )

        results.append(
            {
                "query": query,
                "expected_keywords": expected_keywords,
                f"recall_at_{k}": r_at_k,
                "failure_tags": tags,
                "top_sources": [
                    {
                        "source": ch.source,
                        "score": ch.score,
                        "doc_type": ch.metadata.get("document_type"),
                    }
                    for ch in top_k[:3]
                ],
            }
        )

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    print(f"\n{'='*60}")
    print(f"  Average Recall@{k}: {avg_recall:.3f}  ({len(dataset)} queries)")
    print(f"{'='*60}\n")

    # Write results
    out_path = settings.rag_root / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(dataset_path),
                "k": k,
                f"avg_recall_at_{k}": avg_recall,
                "cases": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation Runner")
    settings = get_settings()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.eval_dataset_path,
        help="Path to eval_dataset.json",
    )
    parser.add_argument(
        "--k", type=int, default=settings.retrieval_final_k, help="Top-K for recall"
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: Dataset not found at {args.dataset}")
        sys.exit(1)

    run_eval(args.dataset, args.k)


if __name__ == "__main__":
    main()
