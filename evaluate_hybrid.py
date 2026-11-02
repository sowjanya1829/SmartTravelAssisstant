
import json
import statistics
import time
from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from hybrid_retriever import (
    COLLECTION_NAME,
    STORAGE_PATH,
    embedding_model,
    load_documents,
    tokenize,
    hybrid_search,
    generate_answer
)
from evals_dataset import eval_dataset


TOP_K = 3
OUTPUT_FILE = "hybrid_eval_results.json"


def normalize_source(source):
    return str(source).replace("\\", "/").strip().lower()


def calculate_metrics(results, relevant_sources):
    relevant_sources = {
        normalize_source(source)
        for source in relevant_sources
    }

    retrieved_sources = [
        normalize_source(item.get("source", "Unknown"))
        for item in results
    ]

    hits = [
        source in relevant_sources
        for source in retrieved_sources
    ]

    precision = sum(hits) / len(hits) if hits else 0.0

    unique_relevant = set(retrieved_sources) & relevant_sources
    recall = (
        len(unique_relevant) / len(relevant_sources)
        if relevant_sources else 0.0
    )

    mrr = 0.0
    for rank, is_relevant in enumerate(hits, start=1):
        if is_relevant:
            mrr = 1 / rank
            break

    return {
        "precision_at_k": precision,
        "source_recall_at_k": recall,
        "mrr": mrr,
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
    client = QdrantClient(path=str(STORAGE_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise ValueError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist."
            )

        print("Loading documents for BM25...")
        documents = load_documents(client)

        tokenized_documents = [
            tokenize((doc.payload or {}).get("content", ""))
            for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_documents)

        all_results = []
        latencies = []
        generation_latencies = []

        for index, case in enumerate(eval_dataset, start=1):
            question = case["question"]
            relevant_sources = case["relevant_sources"]

            print(f"\n{'=' * 60}")
            print(f"Question {index}: {question}")

            start = time.perf_counter()

            results = hybrid_search(
                question=question,
                client=client,
                documents=documents,
                bm25=bm25,
                tokenized_documents=tokenized_documents,
                limit=TOP_K,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Measure Groq answer-generation latency separately
            generation_start = time.perf_counter()

            generated = generate_answer(
                question=question,
                retrieved_documents=results,
            )

            generation_latency_ms = (
                time.perf_counter() - generation_start
            ) * 1000

            generation_latencies.append(generation_latency_ms)

            metrics = calculate_metrics(
                results, relevant_sources
            )

            record = {
                "question": question,
                "relevant_sources": relevant_sources,
                "retrieved_chunks": results,
                "generated_answer": generated["answer"],
                "sources": generated["sources"],
                **metrics,
                "retrieval_latency_ms": round(latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2)
            }

            all_results.append(record)

            for rank, item in enumerate(results, start=1):
                print(
                    f"{rank}. {item['source']} "
                    f"(reranker_score={item['score']:.4f})"
                )

            print(f"Precision@{TOP_K}: {metrics['precision_at_k']:.3f}")
            print(f"Source Recall@{TOP_K}: {metrics['source_recall_at_k']:.3f}")
            print(f"MRR: {metrics['mrr']:.3f}")
            print(f"Retrieval latency: {latency_ms:.2f} ms")
            print(f"Generation latency: {generation_latency_ms:.2f} ms")

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
                statistics.mean(r["mrr"] for r in all_results), 4
            ),
            "retrieval_latency": summarize_latency(latencies),
            "generation_latency": summarize_latency(generation_latencies),
        }

        output = {
            "method": "hybrid_retrieval_with_reranking",
            "summary": summary,
            "results": all_results,
        }

        output_path = Path(OUTPUT_FILE)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\n{'=' * 60}")
        print("HYBRID RETRIEVAL EVALUATION SUMMARY")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved results to: {output_path.resolve()}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
