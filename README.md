# RAG Assistant — Pinecone + Ollama

A fully local, production-oriented Retrieval-Augmented Generation (RAG) application. PDF documents are chunked, embedded, and indexed in Pinecone (serverless), and questions are answered by a locally running Ollama model grounded strictly in retrieved context. No OpenAI APIs or external LLM providers are used.

## Features

- PDF ingestion with page-level provenance tracking
- Recursive chunking with PDF text-artifact cleanup (de-spacing, whitespace normalization)
- Embeddings generated locally via Ollama (`nomic-embed-text`, 768-dim)
- Pinecone serverless vector storage with idempotent, deterministic upserts
- Semantic search with configurable top-K and similarity threshold
- Source-grounded prompt construction with citation markers
- Streaming answer generation via Ollama (`gemma3` or any local model)
- CLI and Streamlit chat interface
- Keyword-based evaluation suite and single-query debug tracing

## Project structure

This reflects the actual files present in the project, not an idealized layout:

```
rag-with-pinecone/
├── .env                         Pinecone + Ollama config
├── requirements.txt
├── app.py                       Streamlit chat UI
├── main.py                      CLI entry point (ingest / ask)
├── validate_config.py           Step 1 — config smoke test
├── validate_pinecone.py         Step 2 — Pinecone connectivity smoke test
├── validate_loader.py           Step 3 — PDF loading smoke test
├── validate_chunker.py          Step 4 — chunking smoke test
├── validate_embedder.py         Step 5 — embedding smoke test
├── validate_upsert.py           Step 6 — full ingestion + upsert smoke test
├── validate_retrieval.py        Step 7 — semantic search smoke test
├── validate_prompt.py           Step 8 — prompt assembly smoke test
├── validate_generation.py       Step 9 — end-to-end generation smoke test
├── validate_eval.py             Step 10 — batch eval / debug runner
└── src/
    ├── config/
    │   ├── __init__.py
    │   ├── settings.py          Typed, fail-fast configuration singleton
    │   └── pinecone_client.py   Pinecone client + serverless index bootstrap
    ├── ingestion/
    │   ├── __init__.py
    │   ├── loader.py            PDF → page-level Document objects
    │   ├── chunker.py           Document → overlapping chunks
    │   ├── upsert.py            Embedded chunks → Pinecone
    │   └── pipeline.py          Orchestrates load → chunk → embed → upsert
    ├── embeddings/
    │   ├── __init__.py
    │   └── embedder.py          Batched embedding generation via Ollama
    ├── retrieval/
    │   ├── __init__.py
    │   └── retriever.py         Query embedding + Pinecone similarity search
    ├── generation/
    │   ├── __init__.py
    │   ├── prompt_builder.py    Context-grounded RAG prompt assembly
    │   ├── generator.py         Streaming generation via Ollama
    │   └── rag.py               ask() — search → prompt → generate
    └── utils/
        ├── __init__.py
        ├── logger.py            Rich-based structured logging
        └── eval.py              Batch evaluation suite + query debugger
```

> **Note on `data/raw/` and `data/processed/`:** these directories hold your source PDFs and are referenced by `settings.data_raw` / `settings.data_processed` in `src/config/settings.py`, but only `data_raw` is actually read by any code path (`load_pdfs_from_dir`). `data_processed` is defined in settings but no current module writes to it — treat it as reserved for future use (e.g., caching parsed chunks), not as an active part of the pipeline today.

### Design decisions

- **Config is a frozen dataclass singleton.** Missing required environment variables raise immediately at import time rather than failing deep inside a request.
- **Vector IDs are deterministic** (`sha256(source + page + chunk_index)`), so re-ingesting the same PDF overwrites existing vectors instead of duplicating them.
- **`eval.py` is intentionally excluded from `src/utils/__init__.py`'s exports.** It depends on the full generation stack (`src.generation.rag.ask`); exporting it from the base `utils` package — which `pinecone_client.py` and other low-level modules import early — creates a circular import. Import it directly: `from src.utils.eval import run_eval, debug_query`.
- **Streamlit ingestion runs synchronously inside `st.spinner`**, not in a background thread. Streamlit's session state isn't reliably observable from outside the main script thread (`missing ScriptRunContext!` warnings), so a direct synchronous call is simpler and correct.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running locally
- A Pinecone account and API key ([app.pinecone.io](https://app.pinecone.io))

Pull the required Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull gemma3
```

Swap `gemma3` for any chat model already available locally (`llama3`, `mistral`, `deepseek-r1`, etc.) by changing `OLLAMA_GENERATION_MODEL` in `.env`. Run `ollama list` to see what's installed.

## Setup

```bash
# 1. Activate your virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file (not committed to version control)
```

Create a `.env` file in the project root with the following keys:

| Variable                  | Description                                  | Default                  |
| ------------------------- | -------------------------------------------- | ------------------------ |
| `PINECONE_API_KEY`        | Your Pinecone API key (required, no default) | —                        |
| `PINECONE_INDEX_NAME`     | Index name (created automatically if absent) | `rag-index`              |
| `PINECONE_CLOUD`          | Serverless cloud provider                    | `aws`                    |
| `PINECONE_REGION`         | Serverless region                            | `us-east-1`              |
| `OLLAMA_BASE_URL`         | Ollama server address                        | `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL`      | Embedding model                              | `nomic-embed-text`       |
| `OLLAMA_GENERATION_MODEL` | Chat/generation model                        | `gemma3`                 |
| `CHUNK_SIZE`              | Max characters per chunk                     | `512`                    |
| `CHUNK_OVERLAP`           | Character overlap between chunks             | `64`                     |
| `RETRIEVAL_TOP_K`         | Default number of chunks retrieved per query | `8`                      |

Only `PINECONE_API_KEY` is required — everything else has a working default defined in `src/config/settings.py`.

## Usage

### 1. Add documents

Drop one or more PDFs into `data/raw/`.

### 2. Ingest

```bash
python main.py ingest
```

This loads PDFs, chunks them, generates embeddings, and upserts vectors into Pinecone. Safe to re-run — existing chunks are overwritten by ID, not duplicated.

### 3. Ask questions

CLI:

```bash
python main.py ask "What gift did Jim buy for Della?"
```

Streamlit UI:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar lets you switch generation models, adjust top-K and similarity threshold, toggle source visibility, and re-run ingestion without leaving the browser.

### 4. Evaluate

```bash
python validate_eval.py          # run the batch eval suite
python validate_eval.py debug    # full retrieval + prompt trace for one query
```

## Validation scripts

Each pipeline stage has a standalone smoke test, useful when modifying any single component in isolation:

| Script                   | Validates                                 |
| ------------------------ | ----------------------------------------- |
| `validate_config.py`     | Environment configuration loads correctly |
| `validate_pinecone.py`   | Pinecone connectivity and index creation  |
| `validate_loader.py`     | PDF text extraction                       |
| `validate_chunker.py`    | Chunking and text cleanup                 |
| `validate_embedder.py`   | Embedding generation via Ollama           |
| `validate_upsert.py`     | Full ingestion pipeline + Pinecone upsert |
| `validate_retrieval.py`  | Semantic search quality                   |
| `validate_prompt.py`     | RAG prompt assembly                       |
| `validate_generation.py` | End-to-end answer generation              |
| `validate_eval.py`       | Batch evaluation + single-query debugging |

## Troubleshooting

**`AttributeError: module 'src.config.settings' has no attribute 'pinecone'`**
Stale `.pyc` cache. Clear it: `rd /s /q src\__pycache__ src\config\__pycache__ src\utils\__pycache__` (Windows) or `find . -name __pycache__ -exec rm -rf {} +` (macOS/Linux), then re-run.

**`ImportError: cannot import name 'X' from partially initialized module` (circular import)**
Check whether `src/utils/__init__.py` re-exports `eval.py`. It shouldn't — import eval utilities directly: `from src.utils.eval import run_eval, debug_query`.

**Ollama connection errors**
Confirm the Ollama server is running: `ollama list` should return without error. If not, start it with `ollama serve` in a separate terminal.

**Streamlit "Run Ingestion" appears stuck**
Resolved by running ingestion synchronously inside `st.spinner` rather than a background thread (current `app.py`). If your local copy still uses `threading.Thread`, replace that block with a direct synchronous call.

## Extending

- **Multiple PDFs**: drop additional files into `data/raw/` and re-run ingestion — `load_pdfs_from_dir` processes the entire directory recursively.
- **Different embedding model**: changing `OLLAMA_EMBED_MODEL` requires re-ingestion, since `EMBEDDING_DIMENSION` in `pinecone_client.py` (currently hardcoded to `768` for `nomic-embed-text`) must match the new model's output dimension.
- **Score-based filtering**: pass `score_threshold` to `search()` to discard low-confidence matches before they reach the prompt — exposed as a slider in the Streamlit sidebar.
- **Larger document sets**: increase `UPSERT_BATCH_SIZE` in `upsert.py` (capped at Pinecone's serverless batch limits) and `DEFAULT_BATCH_SIZE` in `embedder.py` for throughput tuning.
