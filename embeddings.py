
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. Find all city guide files
data_folder = Path("data/cities")
file_paths = sorted(data_folder.glob("*.txt"))

if not file_paths:
    raise FileNotFoundError(
        "No .txt files found in data/cities/"
    )

# 2. Configure the same text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)

all_chunks = []

# 3. Read and split each city's guide
for file_path in file_paths:
    text = file_path.read_text(encoding="utf-8")
    city = file_path.stem.replace("_", " ").title()

    chunks = text_splitter.create_documents(
        texts=[text],
        metadatas=[
            {
                "city": city,
                "source": str(file_path),
            }
        ],
    )

    all_chunks.extend(chunks)

# 4. Load the local embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Convert every chunk into a vector
texts = [chunk.page_content for chunk in all_chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# 6. Check the results
print("\nEmbedding generation successful!")
print("Number of chunks:", len(all_chunks))
print("Embedding array shape:", embeddings.shape)
print("Dimensions per vector:", embeddings.shape[1])

# 7. Display one example
print("\nFirst chunk:")
print(all_chunks[0].page_content)

print("\nFirst chunk metadata:")
print(all_chunks[0].metadata)

print("\nFirst 10 vector values:")
print(embeddings[0][:10])
