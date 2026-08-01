# ChatVeritas

ChatVeritas is a document-grounded AI assistant for answering questions over a
private collection of text documents. It combines a custom two-stage retrieval
system with either a local Qwen model and optional LoRA adapter or a Groq-backed
deployment mode.

The project keeps knowledge outside the language model. Documents are embedded
and indexed locally, relevant passages are retrieved and reranked for each
question, and the selected LLM receives that context in a constrained RAG
prompt. This gives the application a clear separation between document
knowledge, retrieval quality, and text generation.

## Features

- Retrieval-Augmented Generation (RAG) over local text documents
- Sentence Transformer embeddings and FAISS `IndexFlatL2` vector search
- Cross-encoder reranking before context assembly
- Local Qwen 2.5 inference with optional PEFT LoRA adapter loading
- Groq-backed deployment mode using the same retrieval pipeline
- Streamlit interfaces for offline and deployment modes
- Interactive terminal chat for local inference
- LCEL orchestration that preserves the project's custom retrieval and model logic

## Project Architecture

```text
Question
  -> Query embedding
  -> FAISS candidate search
  -> Cross-encoder reranking
  -> Context-grounded prompt
  -> Local Qwen + LoRA or Groq
  -> Response, sources, and timing metrics
```

The retrieval strategy, reranker, prompt wording, and model-loading behavior
remain explicit project components. LangChain Core provides orchestration and
prompt-rendering primitives rather than a replacement RAG stack.

## Technologies Used

- Python, PyTorch, Transformers, and PEFT
- Sentence Transformers and FAISS
- LangChain Core (LCEL, `PromptTemplate`, and output parsing)
- Streamlit
- Groq and the OpenAI-compatible client
- TRL, Datasets, and BitsAndBytes for QLoRA fine-tuning

## Quick Start

### Installation

```powershell
git clone <repository-url>
cd ChatVeritas
python -m pip install -r requirements.txt
```

Create a `.env` file with `GROQ_API_KEY` before using deployment mode.

### Run Offline Streamlit

```powershell
streamlit run app_offline.py
```

Choose whether to load the LoRA adapter from the sidebar.

### Run Deploy Streamlit

```powershell
streamlit run app_deploy.py
```

This mode keeps the same retrieval and reranking pipeline but sends generation
requests to Groq.

### Run Terminal Chat

```powershell
python scripts/chat.py
```

## Repository Structure

```text
ChatVeritas/
├── app_offline.py       # Local Streamlit entry point
├── app_deploy.py        # Groq Streamlit entry point
├── config/              # Application configuration
├── core/                # Configuration, logging, exceptions, runtime setup
├── data/                # Source data and generated runtime artifacts
├── interfaces/          # Stable ChatVeritas application facade
├── llm/                 # Local and deployment generation backends
├── models/              # LoRA adapters and training checkpoints
├── pipelines/           # LCEL runtime orchestration
├── prompts/             # Context-grounded RAG prompt
├── retrieval/           # Embedding, FAISS, and reranking implementation
├── scripts/             # Ingestion, training, chat, and maintenance commands
├── tests/               # Unit and orchestration-contract tests
└── utils/               # Fine-tuning dataset utilities
```

## Documentation

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the complete technical
specification: module contracts, runtime and data pipelines, configuration,
fine-tuning, LangChain integration, engineering decisions, and developer
extension guidance. It is intended for developers who need to understand or
maintain the entire system.

## License

Distributed under the [MIT License](LICENSE).
