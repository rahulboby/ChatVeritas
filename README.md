# ChatVeritas

ChatVeritas is a document-grounded RAG assistant for local knowledge. It uses
local retrieval over a private text corpus and only sends generation to the
selected backend when needed.

The application separates retrieval and generation:

- **Retrieval** is always local: question embedding, FAISS search, and
  cross-encoder reranking.
- **Generation** can be:
  - a local Hugging Face model with optional PEFT LoRA adapter,
  - LM Studio local OpenAI-compatible generation,
  - or Groq cloud generation.

## Key features

- Streamlit multipage UI with Offline Chat, Cloud Chat, and Architecture pages
- Local FAISS retrieval and reranking for grounded answers
- Local model inference with optional PEFT LoRA adapters
- LM Studio local server support for OpenAI-compatible offline generation
- Groq cloud generation for lightweight production experimentation
- Fresh Load sidebar support for clearing cached models and reloading backend
- Utilities for corpus ingestion, chunk inspection, synthetic dataset creation,
  and QLoRA training
- Centralized settings in `configs/settings.py`

## Quick start

### Install

```powershell
git clone <repository-url>
cd ChatVeritas
python -m pip install -r requirements.txt
```

### Configure

All runtime settings live in `configs/settings.py`. Use it to adjust:

- data, cache, and model paths
- local and LM Studio generation settings
- Groq model and API key settings
- retrieval, embedding, and reranking parameters
- synthetic question generation and fine-tuning settings

Use `.env` for secrets. Cloud Chat requires `GROQ_API_KEY`. LM Studio can use
an optional `LMSTUDIO_API_KEY` when the local server requires auth.

### Run

```powershell
streamlit run app.py
```

Then use the sidebar to open:

- **Offline Chat** for Local Model or LM Studio
- **Cloud Chat** for Groq generation
- **Architecture** for the built-in architecture reference

## Corpus and retrieval

Add UTF-8 `.txt` files to `data/raw/` and build the vector store with:

```powershell
python scripts/ingest.py
```

The app uses generated artifacts from `data/vectorstore/` at runtime.
Re-run ingestion only when the corpus changes.

The architecture reference file
`data/raw/chatveritas_architecture.txt` is part of the searchable corpus.
If you update it and want the new content to be retrievable, run
`python scripts/ingest.py` again.

## Fresh Load behavior

The Streamlit pages include a sidebar `Fresh Load` action that clears cached
model resources and reloads the selected backend. Use it when you change the
model, adapter, LM Studio server, or local runtime state.

## Useful commands

```powershell
# Inspect indexed chunks
python scripts/print_chunks.py

# Build synthetic fine-tuning data
python scripts/prepare_finetune_dataset.py

# Validate fine-tuning dataset
python scripts/fine_tune.py --validate-only

# Train a QLoRA adapter
python scripts/fine_tune.py
```

## Repository structure

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

For full architecture and design details, see
[ChatVeritas_ARCHITECTURE.md](ChatVeritas_ARCHITECTURE.md).

## License

Distributed under the [MIT License](LICENSE).
