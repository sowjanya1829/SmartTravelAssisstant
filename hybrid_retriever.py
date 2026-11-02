
import re
from pathlib import Path
import os
from dotenv import load_dotenv
from groq import Groq

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Configuration
COLLECTION_NAME = "travel_knowledge"
STORAGE_PATH = os.getenv("QDRANT_PATH", "qdrant_storage")

load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

DENSE_CANDIDATES = 10
BM25_CANDIDATES = 10
FINAL_CANDIDATES = 3
RRF_K = 60


# 1. Load the existing dense embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load the CrossEncoder reranker
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def tokenize(text):
    """Basic tokenization for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def load_documents(client):
    """Load text and metadata from the existing Qdrant collection."""

    documents = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}

            if payload.get("content"):
                documents.append({
                    "id": record.id,
                    "content": payload.get("content", ""),
                    "city": payload.get("city", ""),
                    "source": payload.get("source", ""),
                })

        if offset is None:
            break

    if not documents:
        raise ValueError(
            "No documents with a 'content' field were found."
        )

    return documents


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Merge ranked lists using Reciprocal Rank Fusion."""

    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = (
                scores.get(document_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


def hybrid_search(
    question,
    client,
    documents,
    bm25,
    tokenized_documents,
    limit=FINAL_CANDIDATES,
):
    """Dense + BM25 retrieval, RRF fusion, and reranking."""

    # Stage A: dense semantic search
    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=DENSE_CANDIDATES,
        with_payload=True,
    ).points

    dense_ranking = [result.id for result in dense_results]

    # Stage B: BM25 keyword search
    query_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:BM25_CANDIDATES]

    bm25_ranking = [
        documents[i]["id"] for i in bm25_indices
        if bm25_scores[i] > 0
    ]

    # Stage C: combine both ranked lists
    fused_ids = reciprocal_rank_fusion(
        [dense_ranking, bm25_ranking]
    )

    document_by_id = {
        document["id"]: document
        for document in documents
    }

    candidates = [
        document_by_id[document_id]
        for document_id in fused_ids
        if document_id in document_by_id
    ]

    if not candidates:
        return []

    # Stage D: rerank query-document pairs
    pairs = [
        (question, document["content"])
        for document in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        {
            "id": document["id"],
            "score": float(score),
            "city": document["city"],
            "source": document["source"],
            "content": document["content"],
        }
        for document, score in reranked[:limit]
    ]




def generate_answer(question, retrieved_documents):
    """Generate an answer from retrieved travel context using Groq."""

    if not retrieved_documents:
        return {
            "answer": (
                "I couldn't find relevant information in the "
                "travel knowledge base to answer this question."
            ),
            "sources": [],
        }

    # Build context from the reranked documents.
    context_parts = []

    for index, document in enumerate(retrieved_documents, start=1):
        content = document.get("content", "").strip()

        if content:
            context_parts.append(
                f"[Source {index} | City: {document['city']} "
                f"| File: {document['source']}]\n{content}"
            )

    if not context_parts:
        return {
            "answer": "The retrieved documents contain no usable text.",
            "sources": [],
        }

    context = "\n\n".join(context_parts)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful travel assistant. "
                    "Answer the user's question using the supplied "
                    "travel context. Do not invent unsupported facts. "
                    "If the context does not contain enough information, "
                    "clearly say so. Keep the answer useful and readable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Travel context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content or ""

    sources = list(dict.fromkeys(
        document["source"]
        for document in retrieved_documents
        if document.get("source")
    ))

    return {
        "answer": answer,
        "sources": sources,
    }



def main():
    client = QdrantClient(path=str(STORAGE_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' does not exist."
            )

        documents = load_documents(client)

        # Build BM25 once, not for every user question.
        tokenized_documents = [
            tokenize(document["content"])
            for document in documents
        ]

        bm25 = BM25Okapi(tokenized_documents)

        print(f"Loaded {len(documents)} text chunks.")
        print("Hybrid search with reranking is ready.")
        print("Type 'exit' to quit.")

        while True:
            question = input("\nYou: ").strip()

            if question.lower() == "exit":
                break

            if not question:
                continue

            results = hybrid_search(
                question,
                client,
                documents,
                bm25,
                tokenized_documents,
            )

            print("\nGenerating answer with Groq...")

            try:
                response = generate_answer(question, results)

                print("\nTravel Assistant:")
                print(response["answer"])

                print("\nSources:")
                for source in response["sources"]:
                    print(f"- {source}")

            except Exception as error:
                print(f"Answer generation failed: {error}")
    finally:
        client.close()




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retrieval resources once at API startup."""
    client = None

    try:
        client = QdrantClient(path=str(STORAGE_PATH))

        documents = load_documents(client)

        if not documents:
            raise RuntimeError(
                "No documents found in the Qdrant collection."
            )

        tokenized_documents = [
            tokenize(doc["content"]) for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_documents)

        app.state.qdrant_client = client
        app.state.documents = documents
        app.state.tokenized_documents = tokenized_documents
        app.state.bm25 = bm25

        print("SmartAgentic Travel API is ready.")
        print(f"Loaded {len(documents)} documents.")

        yield

    finally:
        if client is not None:
            client.close()


app = FastAPI(
    title="SmartAgentic Travel API",
    description="Travel recommendations powered by hybrid retrieval and Groq.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health_check():
    if not hasattr(app.state, "documents"):
        raise HTTPException(
            status_code=503,
            detail="Travel knowledge base is not initialized.",
        )

    return {
        "status": "healthy",
        "documents_loaded": len(app.state.documents),
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        retrieved_documents = hybrid_search(
            question=question,
            client=app.state.qdrant_client,
            documents=app.state.documents,
            bm25=app.state.bm25,
            tokenized_documents=app.state.tokenized_documents,
            limit=3,
        )

        if not retrieved_documents:
            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in the travel knowledge base."
                ),
                "sources": [],
            }

        result = generate_answer(question, retrieved_documents)

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }

    except Exception as exc:
        print(f"Error processing question: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check the server logs.",
        ) from exc



if __name__ == "__main__":
    main()
