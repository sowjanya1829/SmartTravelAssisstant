from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embeddings import embeddings, all_chunks

#define the path to Qdrant database
storage_path=Path("qdrant_storage")
storage_path.mkdir(exist_ok=True)

client=QdrantClient(path=str(storage_path))

#configure the collection
collection_name="travel_knowledge"

#recreate the collection when we rebuild the knowledge base-prevent duplicates
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE
    )
)

#prepare index,chunks and metadata for storage in points
points=[]

for i,(chunk,vector) in enumerate(zip(all_chunks,embeddings)):
    points.append(
        PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "content":chunk.page_content,
                "city":chunk.metadata["city"],
                "source":chunk.metadata["source"]
            }
        )
    )

#store the points into collections
if points:
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )

#verify storage
collection_info=client.get_collection(collection_name)

print(f"Collection {collection_name} created successfully!")
print(f" vectors stored: {collection_info.points_count}")

#release the local database handle
client.close()