# SmartAgentic Travel ✈️

An AI-powered travel assistant that answers questions about destinations using a travel knowledge base, hybrid information retrieval, and a Groq-hosted large language model.

The project combines semantic search and keyword search to retrieve relevant travel information, reranks the results, and uses the retrieved context to generate a natural-language answer with source information.

## Features

- Travel question answering using Retrieval-Augmented Generation (RAG)
- Dense semantic retrieval using Sentence Transformers and Qdrant
- Keyword-based retrieval using BM25
- Reciprocal Rank Fusion (RRF) to combine retrieval rankings
- Cross-Encoder reranking to improve candidate relevance
- Groq LLM answer generation using retrieved context
- FastAPI endpoints for health checks and question answering
- Streamlit user interface
- Docker Compose setup for running the API and UI as separate services

## Architecture

```text
                         ┌────────────────────────┐
                         │       User             │
                         │  Enters travel query   │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │     Streamlit UI       │
                         │     Port: 8501         │
                         └───────────┬────────────┘
                                     │ HTTP request
                                     ▼
                         ┌────────────────────────┐
                         │      FastAPI API       │
                         │      Port: 8000        │
                         │        POST /ask       │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │   Hybrid Retrieval     │
                         └───────────┬────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         ┌────────────────────┐           ┌────────────────────┐
         │   Dense Retrieval  │           │   Sparse Retrieval │
         │ SentenceTransformer│           │       BM25         │
         │      + Qdrant      │           │                    │
         └──────────┬─────────┘           └──────────┬─────────┘
                    └────────────────┬────────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │ Reciprocal Rank Fusion │
                         └───────────┬────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │  Cross-Encoder         │
                         │      Reranking         │
                         └───────────┬────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │      Groq LLM          │
                         │ Context-based answer   │
                         └───────────┬────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │ Answer + Source data   │
                         └────────────────────────┘
```

### Query processing flow

1. The user submits a travel question through Streamlit.
2. Streamlit sends the question to the FastAPI `POST /ask` endpoint.
3. The question is embedded using `sentence-transformers/all-MiniLM-L6-v2`.
4. Qdrant performs dense semantic retrieval.
5. BM25 retrieves documents based on keyword relevance.
6. Reciprocal Rank Fusion combines the two ranked result lists.
7. A Cross-Encoder reranks the retrieved candidates.
8. The selected document context is sent to the Groq LLM.
9. FastAPI returns the generated answer and source information to Streamlit.

## Technology stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Qdrant |
| Keyword retrieval | BM25 (`rank-bm25`) |
| Rank fusion | Reciprocal Rank Fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM inference | Groq API |
| API | FastAPI |
| UI | Streamlit |
| Containerization | Docker and Docker Compose |


## Requirements

- Python 3.11 or a compatible version supported by the installed dependencies
- Git
- Docker Desktop with Docker Compose
- A Groq API key
- Internet access for installing Python dependencies and downloading models the first time they are used

## Setup and run locally

### 1. Clone the repository

```bash
git clone https://github.com/sowjanya1829/SmartTravelAssistant.git
cd SmartTravelAssistant
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_configured_groq_model
```

Replace the example values with your own credentials and model configuration. **Never commit `.env` or expose your API key.**

### 5. Prepare the travel knowledge base

Place your travel text files in `data/cities/`.

Run the ingestion scripts in the order required by your implementation. For the current project, this generally means:

```bash
python split_documents.py
python store_embeddings.py
```

This prepares document chunks, creates embeddings, and stores the travel data in the local Qdrant collection. If the Qdrant collection is already populated, you may not need to run ingestion again.

### 6. Start the FastAPI backend

```bash
uvicorn hybrid_retriever:app --reload --host 127.0.0.1 --port 8000
```

Open the API documentation:

- http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 7. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run streamlit_app.py
```

Open the local Streamlit URL printed in the terminal, usually:

- http://localhost:8501

## Run with Docker

Docker Compose runs the FastAPI backend and Streamlit frontend as separate services.

### 1. Check Docker configuration

Ensure the project root contains:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env`

The `.env` file should contain your Groq settings and must remain untracked by Git.

### 2. Build and start the containers

From the project root:

```bash
docker compose up --build
```

### 3. Open the application

- Streamlit UI: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs
- FastAPI health check: http://localhost:8000/health

The Streamlit container connects to the backend using the Compose service address `http://backend:8000`. From your host browser, use `localhost` URLs instead.

### 4. Stop the containers

```bash
docker compose down
```

The Qdrant storage directory is mounted as persistent data so that stopping the containers does not remove the local knowledge base.

## API endpoints

### `GET /health`

Checks whether the API has initialized its knowledge base.

Example:

```bash
curl http://localhost:8000/health
```

### `POST /ask`

Accepts a travel question and returns a generated answer with source information.

Request:

```json
{
  "question": "What are the best places to visit in Hyderabad?"
}
```

Example response shape:

```json
{
  "answer": "A generated answer based on retrieved travel information.",
  "sources": []
}
```

The exact source entries depend on the documents retrieved for the question.

## Retrieval approach

### Dense retrieval

The query and travel documents are converted into vector embeddings. Qdrant retrieves documents with similar vector representations, helping match queries with related meanings even when they use different wording.

### BM25 retrieval

BM25 performs keyword-based ranking. It is useful when a query contains exact place names or terms that should appear in the retrieved documents.

### Reciprocal Rank Fusion

RRF combines the dense and BM25 rankings. The implementation uses a configurable RRF constant to combine the ranks from both retrieval methods.

### Cross-Encoder reranking

The Cross-Encoder evaluates query-document pairs and reranks the retrieved candidates before they are passed to the language model.

## Evaluation

The project includes retrieval evaluation scripts for metrics such as:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Retrieval latency

Run the evaluation scripts using the dataset and configuration in your local project. Report only measured results; do not assume scores or performance values.

## Security notes

- Keep `.env` out of Git.
- Do not place API keys directly in Python files, Dockerfiles, or `docker-compose.yml`.
- Keep local Qdrant storage and virtual environments out of the repository.
- Rotate your Groq key if it is accidentally exposed.

## Current status

- [x] Hybrid retrieval pipeline
- [x] Groq-based answer generation
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] Local Docker Compose setup
- [ ] Cloud deployment (if/when configured)

## Future improvements

- Deploy the application to a cloud hosting platform
- Add answer-quality and groundedness evaluation
- Improve knowledge-base freshness and source validation
- Add caching and request monitoring
- Expand supported destinations and travel information

## Author

**Sowjanya Budumuru**

GitHub: [@sowjanya1829](https://github.com/sowjanya1829)
