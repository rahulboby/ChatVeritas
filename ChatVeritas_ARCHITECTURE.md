# ChatVeritas Architecture Reference

## Purpose and design

ChatVeritas is a document-grounded Retrieval-Augmented Generation application
built around a local text corpus. It does not rely on model memory for knowledge.
Instead, every question is answered from retrieved passages that are stored in a
FAISS vector index.

The application separates retrieval and generation so the same local retrieval
pipeline can serve both Offline Chat and Cloud Chat. The runtime flow is
intentionally consistent across backends:

```text
question
  -> query embedding
  -> FAISS candidate search
  -> cross-encoder reranking
  -> context assembly
  -> grounded RAG prompt
  -> generation backend
  -> response, evidence chunks, and metrics
```

Generation varies by page only:

- **Offline Chat** can use a local Hugging Face model with optional LoRA,
  or an LM Studio local OpenAI-compatible server.
- **Cloud Chat** uses **Groq** for generation while keeping retrieval local.

Embedding, vector search, and reranking remain local in both modes.

## Repository layout

```text
ChatVeritas/
|-- app.py
|-- configs/
|   |-- __init__.py
|   '-- settings.py
|-- core/
|   |-- chatveritas.py
|   |-- constants.py
|   |-- exceptions.py
|   |-- llm.py
|   |-- logger.py
|   |-- pipelines.py
|   |-- prompts.py
|   |-- retrieval.py
|   '-- utils.py
|-- pages/
|   |-- 1_Offline_Chat.py
|   |-- 2_Cloud_Chat.py
|   '-- 3_Architecture.py
|-- data/
|   |-- raw/
|   |-- vectorstore/
|   |-- processed/
|   '-- cache/
|-- models/
|   |-- adapters/
|   '-- checkpoints/
|-- scripts/
|-- tests/
'-- requirements.txt
```

The project keeps business logic in `core/`, configuration in `configs/`, and
Streamlit UI views in `pages/`.

## Configuration

All runtime and training configuration is centralized in
`configs/settings.py`.
That module defines:

- data, cache, model, and checkpoint paths
- retrieval settings, including embedding and FAISS configuration
- generation settings for local, LM Studio, and Groq backends
- LM Studio endpoint, model, and API key behavior
- Groq model and API key settings
- synthetic question generation settings
- fine-tuning and QLoRA training settings

Secrets are not stored in `settings.py`. `GROQ_API_KEY` should be set in
`.env` or the environment. LM Studio normally uses a local default client key
unless authentication is enabled.

## Core modules

| Module | Responsibility |
|---|---|
| `core/chatveritas.py` | Application facade used by Streamlit pages. It chooses the correct pipeline and returns a uniform result contract. |
| `core/llm.py` | Generation clients for local model, LM Studio, and Groq, plus shared provider resolution and completion helpers. |
| `core/pipelines.py` | Common RAG orchestration and pipeline implementations for offline and cloud modes. |
| `core/retrieval.py` | Embedding, FAISS search, reranking, source tracking, and retrieval metrics. |
| `core/prompts.py` | Context-grounded prompt templates and helpers. |
| `core/utils.py` | Data helpers for caching, paragraph chunking, and synthetic question generation. |
| `core/constants.py` | Runtime defaults and safe environment preparation. |
| `core/logger.py` | Shared logging configuration. |
| `core/exceptions.py` | Domain-specific error types for configuration, retrieval, and model loading. |

LangChain Core is used only for lightweight orchestration and prompt primitives.
It does not replace the project-specific retriever, FAISS store, reranker,
grounding prompt, or LLM clients.

## Streamlit application

The Streamlit app starts from `app.py` and exposes three pages:

| Page | Description |
|---|---|
| Offline Chat | Local model or LM Studio local generation with shared retrieval. |
| Cloud Chat | Groq generation with local FAISS retrieval and reranking. |
| Architecture | Displays this architecture reference in the app. |

## Retrieval contract

At query time, the application returns a stable result object with:

```python
{
  "response": str,
  "chunks": list[dict],
  "metrics": dict,
}
```

Each chunk includes source metadata, chunk text, FAISS distance, FAISS rank,
and rerank score. Metrics include embedding time, retrieval time, reranking time,
generation time, prompt tokens, and counts of candidates and retrieved chunks.

## Corpus and index lifecycle

Knowledge documents are UTF-8 `.txt` files in `data/raw/`. Ingestion creates
generated artifacts in `data/vectorstore/`:

```text
data/raw/*.txt
  -> python scripts/ingest.py
  -> data/vectorstore/index.faiss
  -> data/vectorstore/chunks.pkl
```

ChatVeritas queries the generated index and chunk metadata, not the raw files.
Re-ingest only when the corpus changes. Changing UI pages, switching backends,
or refactoring code does not require rebuilding the index.

The architecture reference in `data/raw/chatveritas_architecture.txt` is part of
the searchable corpus. If its content changes and you want those updates to be
retrievable, re-run `scripts/ingest.py`.

## Fine-tuning lifecycle

Fine-tuning is a separate path from query-time serving:

```text
indexed chunks
  -> python scripts/prepare_finetune_dataset.py
  -> data/processed/train.jsonl
  -> python scripts/fine_tune.py
  -> models/adapters and models/checkpoints
```

The dataset generator validates synthetic questions and caches them before
writing JSONL. Training produces a LoRA adapter that can be loaded by the local
base model.

## Operations

```powershell
# Run the app
streamlit run app.py

# Rebuild retrieval artifacts after corpus updates
python scripts/ingest.py

# Inspect current indexed chunks
python scripts/print_chunks.py

# Create synthetic fine-tuning examples
python scripts/prepare_finetune_dataset.py

# Validate or run fine-tuning
python scripts/fine_tune.py --validate-only
python scripts/fine_tune.py
```

## Troubleshooting

- **Groq requests fail**: verify `GROQ_API_KEY` is available and the model is accessible.
- **LM Studio missing model**: start LM Studio, load the correct model, and set `LMSTUDIO_MODEL_ID`.
- **Updated document not found**: run `scripts/ingest.py` after changing `data/raw/`.
- **Local model too large**: reduce memory settings, use LM Studio, or use Cloud Chat with Groq.
- **LoRA adapter fails**: confirm the adapter matches the configured base model.
