from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

#connect to the existing local qdrant database
storage_path=Path("qdrant_storage")

if not storage_path.exists():
    raise FileNotFoundError(
        "Qdrant database not found run store_embeddings.py first"
    )

client=QdrantClient(path=str(storage_path))
collection_name="travel_knowledge"



if not client.collection_exists(collection_name):
    raise ValueError(
        f"Collection {collection_name} not found in the database."
    )
#load the embedding model
model=SentenceTransformer(
    "sentence-transformers/all-miniLM-L6-v2"
)

#define the search function

def search_travel_knowledge(question, limit=3):
    # Convert the user's question into a vector
    query_vector = model.encode(
        question,
        normalize_embeddings=True,
    )

    # Find the most similar vectors in Qdrant
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    ).points

    # Display the retrieved chunks
    if not results:
        print("No matching information found.")
        return

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(results)} chunks\n")

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        print("=" * 50)
        print(f"Result {rank}")
        print(f"Similarity score: {result.score:.4f}")
        print(f"City: {payload.get('city', 'Unknown')}")
        print(f"Source: {payload.get('source', 'Unknown')}")
        print(f"Text: {payload.get('content', '')}")
        print()


# 4. Try asking questions interactively
if __name__ == "__main__":
    try:
        while True:
            query = input(
                "\nAsk about a destination (or type 'exit'): "
            ).strip()

            if query.lower() == "exit":
                break

            if not query:
                continue

            search_travel_knowledge(query)

    finally:
        client.close()
