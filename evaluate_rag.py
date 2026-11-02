
# evaluate_rag.py

import json
import statistics
import time
from pathlib import Path
import re

from rag import (
    client,
    collection_name,
    embedding_model,
    generate_answer,
)
from evals_dataset import eval_dataset


TOP_K = 3
OUTPUT_FILE = "rag_eval_results.json"


def retrieve_baseline(question, limit=TOP_K):
    """Use the existing dense-vector retrieval in Qdrant."""
    start = time.perf_counter()

    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    results = []
    for point in response.points:
        payload = point.payload or {}

        results.append({
            "id": str(point.id),
            "source": payload.get("source", "Unknown"),
            "city": payload.get("city", "Unknown"),
            "content": payload.get("content", ""),
            "score": float(point.score),
        })

    return results, elapsed_ms

def normalize_source(source):
    # Convert Windows backslashes to forward slashes
    source = source.replace("\\", "/")

    # Remove duplicate slashes and normalize capitalization
    source = re.sub(r"/+", "/", source).strip().lower()

    return source

def calculate_retrieval_metrics(results, relevant_sources):
    # Normalize expected and retrieved paths before comparing
    relevant_sources = {
        normalize_source(source)
        for source in relevant_sources
    }

    retrieved_sources = [
        normalize_source(result["source"])
        for result in results
    ]

    hits = [
        source in relevant_sources
        for source in retrieved_sources
    ]

    relevant_hits = sum(hits)

    precision = (
        relevant_hits / len(results)
        if results else 0.0
    )

    unique_retrieved = set(retrieved_sources)
    retrieved_relevant_sources = (
        unique_retrieved & relevant_sources
    )

    recall = (
        len(retrieved_relevant_sources) / len(relevant_sources)
        if relevant_sources else 0.0
    )

    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(hits, start=1):
        if is_relevant:
            reciprocal_rank = 1 / rank
            break

    return {
        "precision_at_k": precision,
        "source_recall_at_k": recall,
        "mrr": reciprocal_rank,
        "retrieved_sources": retrieved_sources,
    }

def summarize_latency(values):
    if not values:
        return {}

    ordered = sorted(values)

    return {
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(
            ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            2,
        ),
    }


def main():
    all_results = []

    retrieval_times = []
    generation_times = []
    end_to_end_times = []

    for index, case in enumerate(eval_dataset, start=1):
        question = case["question"]
        relevant_sources = case["relevant_sources"]

        print(f"\n{'=' * 60}")
        print(f"Question {index}: {question}")

        end_start = time.perf_counter()

        # Measure baseline retrieval
        retrieved, retrieval_ms = retrieve_baseline(
            question,
            limit=TOP_K,
        )





        metrics = calculate_retrieval_metrics(
            retrieved,
            relevant_sources,
        )

        # Measure answer generation separately.
        # generate_answer() performs its own retrieval too.
        generation_start = time.perf_counter()
        answer = generate_answer(question)
        generation_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        end_to_end_ms = (
            time.perf_counter() - end_start
        ) * 1000

        retrieval_times.append(retrieval_ms)
        generation_times.append(generation_ms)
        end_to_end_times.append(end_to_end_ms)

        result = {
            "question": question,
            "relevant_sources": relevant_sources,
            "retrieved_chunks": retrieved,
            **metrics,
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "answer_generation_call_latency_ms": round(
                generation_ms, 2
            ),
            "evaluation_wall_time_ms": round(
                end_to_end_ms, 2
            ),
            "answer": answer,
        }

        all_results.append(result)

        print("\nRetrieved sources:")
        for rank, item in enumerate(retrieved, start=1):
            print(
                f"{rank}. {item['source']} "
                f"(score={item['score']:.4f})"
            )

        print(f"\nPrecision@{TOP_K}: {metrics['precision_at_k']:.3f}")
        print(f"Source Recall@{TOP_K}: {metrics['source_recall_at_k']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")
        print(f"Retrieval latency: {retrieval_ms:.2f} ms")
        print(f"Answer call latency: {generation_ms:.2f} ms")
        print(f"Answer:\n{answer}")

    # Aggregate evaluation metrics
    summary = {
        "number_of_questions": len(all_results),
        "top_k": TOP_K,
        "mean_precision_at_k": round(
            statistics.mean(
                r["precision_at_k"] for r in all_results
            ), 4
        ),
        "mean_source_recall_at_k": round(
            statistics.mean(
                r["source_recall_at_k"] for r in all_results
            ), 4
        ),
        "mean_mrr": round(
            statistics.mean(r["mrr"] for r in all_results),
            4,
        ),
        "retrieval_latency": summarize_latency(retrieval_times),
        "answer_generation_call_latency": summarize_latency(
            generation_times
        ),
        "evaluation_wall_time": summarize_latency(end_to_end_times),
    }

    output = {
        "method": "baseline_dense_retrieval",
        "summary": summary,
        "results": all_results,
    }

    output_path = Path(OUTPUT_FILE)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("BASELINE RAG EVALUATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()
