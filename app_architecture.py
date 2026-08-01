import streamlit as st
import streamlit_mermaid as stm

st.set_page_config(
    page_title="ChatVeritas Architecture",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(

    """
# ChatVeritas Architecture

> This document is the technical specification for ChatVeritas. It describes
> the repository as implemented: component boundaries, data contracts,
> artifacts, configuration, execution paths, and extension points. For a
> concise project introduction and launch commands, see [README.md](README.md).

## 1. Project Overview

ChatVeritas is a document-grounded Retrieval-Augmented Generation (RAG)
application. It answers questions against a local corpus of UTF-8 text files
instead of treating the language model as the source of project knowledge.
The corpus is converted into embeddings and stored in a FAISS index; every
question retrieves and reranks evidence before a response is generated.

The application addresses two practical deployment cases with one retrieval
system:

- **Offline mode** runs `Qwen/Qwen2.5-3B-Instruct` locally and can apply a
  PEFT LoRA adapter.
- **Deploy mode** uses the Groq OpenAI-compatible endpoint for generation when
  hosting the local model is impractical.

The design favors explicit ownership over opaque framework abstractions.
`retrieval/` owns embeddings, FAISS search, and cross-encoder reranking;
`llm/` owns backend-specific loading and generation; `prompts/` owns wording;
and `pipelines/` connects those pieces. LangChain Core is used only for
orchestration and prompt rendering, not to replace the retrieval algorithm or
model backends.

## 2. High-Level Architecture

### System view
"""
)
stm.st_mermaid(
    """
flowchart TD
    U[User] --> UI{Entry point}
    UI -->|Offline Streamlit| AO[app_offline.py]
    UI -->|Deploy Streamlit| AD[app_deploy.py]
    UI -->|Terminal| TC[scripts/chat.py]
    AO --> CV[interfaces.ChatVeritas]
    AD --> CV
    TC --> CV
    CV --> P{Pipeline mode}
    P -->|offline| OP[OfflinePipeline]
    P -->|deploy| DP[DeployPipeline]
    OP --> RP[RAGPipeline LCEL orchestration]
    DP --> RP
    RP --> R[Custom Retriever]
    R --> E[SentenceTransformer embedding]
    E --> F[FAISS IndexFlatL2]
    F --> CE[CrossEncoder reranker]
    CE --> PT[PromptTemplate]
    PT --> L{LLM backend}
    L -->|offline| Q[Qwen + optional LoRA]
    L -->|deploy| G[Groq API]
    Q --> OUT[Response, chunks, metrics]
    G --> OUT
"""
)

st.markdown(
    """
All user-facing paths converge on `ChatVeritas.ask(question)`. The interface
selects a mode-specific pipeline lazily; that pipeline uses the common LCEL
chain in `RAGPipeline` and returns the same public result shape regardless of
backend.

### Runtime data contract

The only result contract exposed beyond `pipelines/` is:

```python
{
    "response": str,
    "chunks": list[dict],
    "metrics": dict,
}
```

Each returned chunk contains `chunk`, `source`, `chunk_id`, `distance`,
`faiss_rank`, and `rerank_score`. `metrics` combines retrieval timing and
metadata with generation timing and prompt-token count. The Streamlit apps use
this information to display the answer, evidence, sources, and diagnostic
timings.

### LCEL chain
"""
)

stm.st_mermaid(
    """
flowchart LR
    I["{'question': question}"]
    R[RunnableLambda: custom retrieve and rerank]
    C[RunnableLambda: join selected chunks]
    P[PromptTemplate]
    L[LLM Runnable: call existing generate]
    O[StrOutputParser]
    F[Format existing result contract]

    I --> R --> C --> P --> L --> O --> F
"""
)

st.markdown(
    """
`RunnablePassthrough.assign` retains the retrieval output while it adds
`context`, `prompt`, `generation`, and `response`. This is important: the
chain does not turn the retrieved chunks into generic LangChain documents or
discard their project-specific metrics.

## 3. Repository Structure

| Path | Why it exists | What belongs there | Primary users |
| --- | --- | --- | --- |
| `app_offline.py` | Starts the local web app. | Streamlit UI and resource caches for offline inference. | Local end users. |
| `app_deploy.py` | Starts the hosted-generation web app. | Streamlit UI and `.env` loading for Groq mode. | Deploy users. |
| `config/` | Separates settings from code. | `config.json` is active; `config_old.json` is not loaded by runtime code. | All pipelines and scripts. |
| `core/` | Holds application-wide concerns. | Config loading, logging, exceptions, and early thread limits. | Entry points and runtime modules. |
| `data/` | Holds user data and generated artifacts. | Raw documents, FAISS artifacts, processed JSONL, cache, and model offload data. | Ingestion, dataset preparation, retrieval, offline loading. |
| `interfaces/` | Preserves a stable application API. | The `ChatVeritas` facade. | Apps and terminal chat. |
| `llm/` | Encapsulates inference-provider behavior. | Base contract, LCEL adapter, local Qwen/LoRA, and Groq backend. | Pipelines. |
| `models/` | Separates trained artifacts from source code. | LoRA adapter, tokenizer files, and training checkpoints. | Fine-tuning and offline inference. |
| `pipelines/` | Owns execution order and backend selection. | Shared LCEL RAG pipeline plus offline/deploy specializations. | `ChatVeritas`. |
| `prompts/` | Gives prompt wording one owner. | The RAG prompt template and render helper. | `RAGPipeline`. |
| `retrieval/` | Keeps the retrieval strategy explicit and reusable. | FAISS retrieval and cross-encoder reranking. | `RAGPipeline`. |
| `scripts/` | Contains executable maintenance and lifecycle commands. | Ingestion, data preparation, training, terminal chat, inspection, scraping. | Developers/operators. |
| `tests/` | Captures behavioral contracts. | Unit tests for utilities, training validation, and LCEL orchestration. | Developers and CI. |
| `utils/` | Reusable fine-tuning support code. | Cache, Groq question generator, paragraph chunking utility. | Dataset preparation and tests. |

The `__init__.py` files only mark importable application directories; they do
not expose a second public architecture.

## 4. Module Documentation

This section documents every executable application module. Package marker
files and tests are omitted because they contain no runtime implementation.

### Entry points and interface

| Module | Purpose and public API | Inputs / outputs / artifacts | Dependencies and call graph | Configuration and extension rationale |
| --- | --- | --- | --- | --- |
| `app_offline.py` | Local Streamlit UI. `get_config()` caches config; `load_chatbot(use_lora)` caches `ChatVeritas`. | Reads configuration and the artifacts loaded by the offline pipeline; stores chat history in `st.session_state`; displays response, chunks, sources, and timings. | Calls `apply_thread_limits`, `load_config`, and `ChatVeritas(mode='offline')`. It is launched by Streamlit, not imported by other application modules. | Reads model labels for the sidebar. It remains separate from `app_deploy.py` because LoRA selection and local loading are offline-specific. |
| `app_deploy.py` | Deploy Streamlit UI. `get_config()` and `load_chatbot()` mirror the offline entry point. | Loads `.env`, reads config, and displays the same result contract. It does not write project artifacts. | Calls `apply_thread_limits`, `load_dotenv`, `load_config`, and `ChatVeritas(mode='deploy')`. | Requires `GROQ_API_KEY` through `DeployLLM`. Keeping it separate keeps hosted credentials and deployment messaging out of local mode. |
| `scripts/chat.py` | Interactive local terminal chat. Public function: `main()`. | Prompts for LoRA use and user questions; prints answers; no persistent output. | Calls `apply_thread_limits` before constructing `ChatVeritas(mode='offline')`. | It intentionally contains no retrieval or model logic, making terminal behavior match the offline app. |
| `interfaces/chatveritas.py` | Defines the stable facade `ChatVeritas(mode, use_lora, config)` and `ask(question)`. | Accepts a non-empty question; returns the common result dictionary. Empty questions return empty response/chunks/metrics without constructing new state. | Called by both apps and terminal chat. Lazily imports `OfflinePipeline` or `DeployPipeline`; calls their `run`. | Uses all runtime configuration indirectly. The facade isolates UIs from LCEL, retrieval, and LLM details and is the right place for future non-UI callers. |

### Core modules

| Module | Purpose and public API | Inputs / outputs / artifacts | Dependencies and call graph | Configuration and extension rationale |
| --- | --- | --- | --- | --- |
| `core/config.py` | `load_config()` is the single configuration loader. | Reads `config/config.json`; returns a Python dictionary. It writes nothing. | Called by entry points, `ChatVeritas` when no config is injected, and lifecycle scripts. | Centralizing the project-root path prevents working-directory-dependent configuration behavior. |
| `core/constants.py` | Defines `THREAD_LIMIT_ENV_VARS` and `apply_thread_limits()`. | Sets environment defaults for OpenMP, MKL, OpenBLAS, NumExpr, tqdm, and Transformers verbosity. | Called before heavy imports by the apps and terminal chat. | Separate because import order matters: NumPy, FAISS, Sentence Transformers, and Transformers may initialize thread pools during import. |
| `core/logger.py` | `get_logger(name)` returns a configured named logger. | Writes formatted logs to stdout; avoids duplicate handlers for the same logger. | Used throughout runtime modules. | A centralized formatter makes instrumentation consistent and can later be extended with file or structured logging. |
| `core/exceptions.py` | Defines `ChatVeritasError`, `ModelLoadError`, `RetrievalError`, and `ConfigurationError`. | No I/O. These types communicate expected failure categories. | Raised by the interface, offline loader, and retriever. | Keeps caller-facing errors distinct from third-party exceptions. |

### Retrieval and prompt modules

| Module | Purpose and public API | Inputs / outputs / artifacts | Dependencies and call graph | Configuration and extension rationale |
| --- | --- | --- | --- | --- |
| `retrieval/retriever.py` | `Retriever(...)` loads the index, chunks, embedding model, and reranker. `retrieve(query)` executes two-stage retrieval. | Reads `index.faiss` and `chunks.pkl`. Returns `{'results': list[chunk], 'metrics': dict}`. It validates index/chunk count equality and rejects an empty index. | Created by `RAGPipeline`; calls SentenceTransformer, FAISS, NumPy, and `Reranker.rerank`. | Uses `embedding`, `reranker`, `retrieval.top_k`, and `retrieval.faiss_candidates`. It is separate so index format, metrics, and retrieval policy remain independent of orchestration. |
| `retrieval/reranker.py` | `Reranker(model_name, device)` loads a CrossEncoder; `rerank(query, candidates, top_k)` scores and sorts candidates. | Mutates candidates by adding `rerank_score`; returns the highest-scoring `top_k` candidates. No files are written. | Constructed by `Retriever`; called after FAISS candidate retrieval. | Uses `reranker.model` and `reranker.device`. It remains custom because it preserves the candidate metadata and ranking policy produced by `Retriever`. |
| `prompts/templates.py` | Holds `RAG_PROMPT_TEMPLATE` and `build_rag_prompt(question, context)`. | Accepts a question and joined context; returns the exact prompt string used by either backend. | Used by `RAGPipeline`; depends on LangChain Core `PromptTemplate`. | There are no prompt-related config keys. The module separates wording from control flow and is the only place to edit the RAG instructions. |

### LLM modules

| Module | Purpose and public API | Inputs / outputs / artifacts | Dependencies and call graph | Configuration and extension rationale |
| --- | --- | --- | --- | --- |
| `llm/base.py` | Abstract `BaseLLM` defines `generate(prompt) -> (response, metrics)` and `as_runnable()`. | No I/O itself. `as_runnable()` preserves the same text-plus-metrics contract for LCEL. | Implemented by `OfflineLLM` and `DeployLLM`; consumed by `RAGPipeline`. | The common contract makes backend replacement possible without changing the interface or result shape. |
| `llm/runnables.py` | Defines protocol `SupportsGeneration` and `create_generation_runnable(llm)`. | Wraps `generate`; emits `{'response': str, 'metrics': dict}`. | Called through `BaseLLM.as_runnable()` by the pipeline. | It is separate so LangChain-specific adapter code does not leak into either model backend. |
| `llm/offline.py` | `OfflineLLM(config, use_lora)` loads and runs local Qwen. `generate(prompt)` applies the tokenizer chat template, calls `model.generate`, decodes only newly generated tokens, and returns generation time and input-token count. | Reads a local adapter if complete; otherwise resolves the configured Hugging Face adapter ID. May create `data/model_offload/`. Does not alter model weights. | Created by `OfflinePipeline`; uses PyTorch, Transformers, PEFT, and `BaseLLM`. | Uses `model`, `generation`, and optional `inference` settings. Separate helper methods make adapter discovery, tokenizer fallback, memory-aware model loading, and LoRA application individually maintainable. |
| `llm/deploy.py` | `DeployLLM(config)` initializes an OpenAI client for Groq; `generate(prompt)` sends system and user messages to the configured model. | Reads `GROQ_API_KEY` from environment; returns response, wall-clock generation time, and provider prompt-token count. No files are written. | Created by `DeployPipeline`; uses OpenAI client and `BaseLLM`. | Uses `generation.model` when present, otherwise `openai/gpt-oss-120b`, plus generation temperature and token limit. It remains custom to retain the Groq endpoint, system message, and metrics semantics. |

### Pipeline modules

| Module | Purpose and public API | Inputs / outputs / artifacts | Dependencies and call graph | Configuration and extension rationale |
| --- | --- | --- | --- | --- |
| `pipelines/rag_pipeline.py` | Abstract `RAGPipeline` owns `run(question)` and the common LCEL chain. Subclasses implement `_create_llm()`. | Reads retrieval artifacts through `Retriever`; returns the common result contract. It writes no project artifacts. | Called by `ChatVeritas` through a concrete pipeline. It calls `Retriever.retrieve`, `PromptTemplate`, the LLM runnable, and `StrOutputParser`. | Uses retrieval, embedding, reranker, paths, and backend generation config. It is separate to eliminate duplicated offline/deploy orchestration without concealing custom retrieval. |
| `pipelines/offline_pipeline.py` | `OfflinePipeline(config, use_lora)` specializes `RAGPipeline`. | Creates the common retriever and an `OfflineLLM`; uses the base `run`. | Built by `ChatVeritas` for offline mode. | Adds only the LoRA flag. Keeping it thin makes the difference from deploy mode explicit. |
| `pipelines/deploy_pipeline.py` | `DeployPipeline(config)` specializes `RAGPipeline`. | Creates the common retriever and a `DeployLLM`; uses the base `run`. | Built by `ChatVeritas` for deploy mode. | Has no extra persistent artifacts. It differs only in generation backend. |

### Data, training, and maintenance modules

| Module | Purpose and public API | Inputs / outputs / artifacts | Dependencies and call graph | Configuration and extension rationale |
| --- | --- | --- | --- | --- |
| `scripts/ingest.py` | `clean_text(text)` removes separator-only lines and extra blank lines. `main()` builds the vector store. | Reads `data/raw/*.txt`; writes `data/vectorstore/index.faiss` and `data/vectorstore/chunks.pkl`. Each chunk stores `chunk_id`, `source`, and `chunk`. | Executed manually; runtime `Retriever` later consumes both artifacts. Uses SentenceTransformer, FAISS, NumPy, and `RecursiveCharacterTextSplitter`. | Uses `paths.raw_data`, `paths.vectorstore`, `embedding`, and `retrieval.chunk_size/chunk_overlap`. It stays a script because indexing is an explicit offline lifecycle step. |
| `scripts/prepare_finetune_dataset.py` | `main()` generates conversational training records from indexed chunks. | Reads `chunks.pkl`; calls Groq through `QuestionGenerator`; optionally reads/writes per-chunk cache JSON; writes JSONL to `paths.processed_data`. | Executed after ingestion; `fine_tune.py` consumes its output. | Uses `llm`, `cache`, and `paths` settings. It deliberately reads exact indexed chunks so training examples stay aligned with inference evidence. |
| `scripts/fine_tune.py` | Public helpers validate data/configuration and split by chunk; `main()` trains and saves a QLoRA adapter. | Reads processed JSONL; writes checkpoints, adapter weights, tokenizer, and `chatveritas_training.json`; can upload to Hugging Face. | Executed manually; `OfflineLLM` consumes the saved adapter. Uses Datasets, Transformers, PEFT, TRL, BitsAndBytes, and Hugging Face Hub lazily. | Uses `model`, `paths`, `training`, `lora`, and optional `hub` settings. Helpers are separate for fast validation and unit testing before GPU loading. |
| `scripts/print_chunks.py` | `main()` prints indexed chunk metadata and text. | Reads `chunks.pkl`; writes only console output. | Manual inspection tool after ingestion. | Uses `paths.vectorstore`. Keeping it separate avoids debug output in runtime retrieval. |
| `scripts/scrape_urls.py` | `sanitize_filename`, `scrape_url`, and `main()` fetch selected URLs and save extracted Markdown-like text. | Reads the hard-coded `URLS`; writes `.txt` files under `data/raw/`. | Optional pre-ingestion helper; uses Trafilatura. | `URLS` and `OUTPUT_DIR` are module constants, not `config.json` keys. It is intentionally opt-in because source selection is editorial. |
| `utils/cache.py` | `CacheManager` provides SHA-256-keyed JSON `get_or_create`, plus read/write/count/clear helpers. | Reads/writes one JSON file per source chunk; uses atomic `os.replace` after a temporary file. | Created by dataset preparation when caching is enabled. | Cache location and enablement come from `cache`. Separate caching avoids repeated paid API calls and supports resuming interrupted generation. |
| `utils/question_generator.py` | `QuestionGenerator` calls Groq to generate a topic and up to five validated questions for a paragraph/chunk. | Takes source text; returns `{'topic': str, 'questions': list[str]}`. It strips, deduplicates, limits, and validates response JSON. | Created by dataset preparation. | Uses `llm.model`, `llm.temperature`, and `llm.max_retries`. It is separate from the RAG deployment backend because synthetic-data generation has a different prompt and response contract. |
| `utils/paragraph_chunker.py` | `ParagraphChunker` token-counts with a Transformers tokenizer and splits paragraphs by sentence boundaries, with a word/token fallback for oversized sentences. | Takes raw text; returns `list[str]`; no artifacts itself. | Covered by tests. It is not called by the current `scripts/ingest.py`, which uses `RecursiveCharacterTextSplitter`. | Constructor accepts model name, `max_tokens` (default 450), and `min_paragraph_length` (default 40). It remains a reusable paragraph-aware utility without silently changing the active ingestion format. |

## 5. Runtime Pipelines

### Offline pipeline
"""
)
stm.st_mermaid(
    """
sequenceDiagram
    participant User
    participant UI as app_offline.py or scripts/chat.py
    participant Interface as ChatVeritas
    participant Pipeline as OfflinePipeline / RAGPipeline
    participant Retriever
    participant Reranker
    participant Model as Qwen + optional LoRA

    User->>UI: question
    UI->>Interface: ask(question)
    Interface->>Pipeline: run(question)
    Pipeline->>Retriever: retrieve(question)
    Retriever->>Reranker: rerank FAISS candidates
    Reranker-->>Pipeline: top-k chunks + retrieval metrics
    Pipeline->>Model: generate(rendered prompt)
    Model-->>Pipeline: response + generation metrics
    Pipeline-->>Interface: response, chunks, metrics
    Interface-->>UI: result
"""
)

st.markdown(
"""
The offline Streamlit app applies thread limits before heavy imports, caches the
configuration and chatbot resource, and exposes a LoRA toggle. The terminal
script similarly applies limits but presents an interactive REPL. Neither
alters retrieval or prompt construction.

### Deploy pipeline
"""
)

stm.st_mermaid(
    """
sequenceDiagram
    participant User
    participant UI as app_deploy.py
    participant Interface as ChatVeritas
    participant Pipeline as DeployPipeline / RAGPipeline
    participant Retriever
    participant Groq as Groq API

    User->>UI: question
    UI->>Interface: ask(question)
    Interface->>Pipeline: run(question)
    Pipeline->>Retriever: retrieve and rerank
    Retriever-->>Pipeline: top-k chunks + retrieval metrics
    Pipeline->>Groq: completion with system message and RAG prompt
    Groq-->>Pipeline: response + usage
    Pipeline-->>UI: response, chunks, metrics
"""
)

st.markdown(
"""
Deploy mode adds `.env` loading and requires `GROQ_API_KEY`; otherwise, its
context and evidence path is identical to offline mode. This is why a response
from either mode can be displayed by the same UI code.

### Terminal pipeline

`scripts/chat.py` is not a separate retrieval pipeline. It asks the operator
whether to use LoRA, creates the same offline `ChatVeritas` instance, then
calls `ask` repeatedly until `exit`, EOF, or Ctrl+C. It does not maintain
conversation memory: each question is an independent RAG invocation.

## 6. Data Pipeline
"""
)

stm.st_mermaid(
    """
flowchart TD
    A[Manual document cleaning] --> B[data/raw/*.txt]
    B --> C[scripts/ingest.py]
    C --> D[Clean text + configured character splitter]
    D --> E[SentenceTransformer document embeddings]
    E --> F[data/vectorstore/index.faiss]
    D --> G[data/vectorstore/chunks.pkl]
    G --> H[scripts/prepare_finetune_dataset.py]
    H --> I[Groq question generation + optional cache]
    I --> J[data/processed/train.jsonl]
    J --> K[scripts/fine_tune.py]
    K --> L[models/checkpoints/]
    K --> M[models/adapters/]
    F --> N[Runtime Retriever]
    G --> N
    M --> O[OfflineLLM]
    N --> P[Inference]
    O --> P
"""
)
st.markdown(
    """

| Stage | Producer | Artifact | Consumer | Notes |
| --- | --- | --- | --- | --- |
| Source preparation | Developer or `scrape_urls.py` | `data/raw/*.txt` | `ingest.py` | Manual cleaning is intentionally outside code because corpus quality is a content decision. |
| Ingestion | `ingest.py` | `data/vectorstore/index.faiss` | `Retriever` | Flat L2 index containing float32 document embeddings. |
| Ingestion | `ingest.py` | `data/vectorstore/chunks.pkl` | `Retriever`, `prepare_finetune_dataset.py`, `print_chunks.py` | Pickled list mapping index position to chunk text, source, and ID. |
| Synthetic data | `prepare_finetune_dataset.py` | `data/cache/*.json` | Dataset preparation itself | Optional cache of validated Groq responses keyed by chunk hash. |
| Synthetic data | `prepare_finetune_dataset.py` | `data/processed/train.jsonl` | `fine_tune.py` | Conversational records with source/chunk provenance. |
| Training | `fine_tune.py` | `models/checkpoints/` | Training resume | Intermediate checkpoints; ignored by Git. |
| Training | `fine_tune.py` | `models/adapters/` | `OfflineLLM` | PEFT adapter, tokenizer, and training summary. |

The active ingestion step uses `RecursiveCharacterTextSplitter` with
paragraph, newline, sentence, word, and character fallbacks. The independent
`ParagraphChunker` utility is token-aware and sentence-aware but is not wired
into the current ingestion command; documentation must not conflate the two.

## 7. Retrieval System

### Embedding and index compatibility

At ingestion, `SentenceTransformer(config['embedding']['model'])` encodes all
chunk text. The array is cast to `float32`, added to `faiss.IndexFlatL2`, and
written to disk. At query time, `Retriever` loads the same configured model,
encodes the question, casts its vector to `float32`, and queries the index.

The document and query embedding models must remain compatible. Changing
`embedding.model` requires rerunning ingestion; otherwise, the stored vectors
and query vectors inhabit different spaces.

### Two-stage ranking

1. **Candidate generation:** FAISS searches for
   `min(retrieval.faiss_candidates, index.ntotal)` nearest vectors by squared
   L2 distance. `IndexFlatL2` is a flat exhaustive index, not an approximate
   index despite the generic term “ANN” sometimes used in retrieval prose.
2. **Reranking:** `Reranker` creates `(question, chunk)` pairs for every FAISS
   candidate. Sentence Transformers `CrossEncoder.predict` scores all pairs,
   appends `rerank_score`, sorts descending, and returns `retrieval.top_k`.
3. **Context:** `RAGPipeline` joins only the final reranked chunk text with
   blank lines. It does not inject source names or scores into the LLM prompt.
   Sources and scores remain available to the UI through the result chunks.

The `Retriever` records embedding, FAISS, and reranking time separately. It
also reports the actual candidate count, returned chunk count, average FAISS
distance, and distinct source filenames.

### Why the system is structured this way

Dense vector search makes broad candidate selection inexpensive for the local
index. The cross-encoder then applies query-document attention to a much
smaller set, improving the final ordering without scoring every corpus chunk.
Keeping this logic in project code protects the exact metadata, timing, and
ranking behavior that the UI and tests expect.

## 8. Fine-Tuning System

### Dataset preparation

Dataset preparation intentionally starts from `chunks.pkl`, not a separately
re-chunked corpus. For every chunk of at least 20 non-whitespace characters,
`QuestionGenerator` asks Groq for one topic and up to five diverse questions
that are answerable from that chunk. Each generated question produces a record
whose assistant response is the original chunk text:

```json
{
  "topic": "...",
  "source_file": "source.txt",
  "chunk_id": 12,
  "messages": [
    {"role": "user", "content": "Generated question"},
    {"role": "assistant", "content": "Original indexed chunk"}
  ]
}
```

`CacheManager.get_or_create` validates cached JSON before reusing it. Invalid,
corrupt, or schema-incompatible cache entries are regenerated and overwritten.

### Training

`scripts/fine_tune.py` first validates the JSONL schema, normalizes message
whitespace, and groups examples by `source_file:chunk_id`. It splits whole
groups into train and validation partitions so multiple generated questions for
the same target chunk do not leak across partitions.

For a real training run, it requires CUDA, loads the Qwen base model in 4-bit
NF4 mode, prepares it for k-bit training, and constructs a causal-LM LoRA
configuration. TRL `SFTTrainer` receives only the `messages` field. The script
saves checkpoints during training, then saves the final PEFT adapter and
tokenizer to the configured adapter directory.

`--validate-only` is the supported fast check: it validates configuration and
dataset content without importing the GPU training stack or loading a model.
`--resume` optionally resumes the newest or named checkpoint.

### Inference relationship

Offline inference checks whether `model.adapter_path` contains a complete
adapter configuration and weight file. If not, it falls back to
`model.adapter_repo_id`. If loading an adapter fails, it logs a warning and
uses the base model instead. Fine-tuning changes generation behavior; RAG
knowledge remains in FAISS and `chunks.pkl`.

## 9. LangChain Integration

LangChain Core was introduced as an orchestration refactor, not as a new RAG
implementation. The application deliberately does **not** use `RetrievalQA`,
`ConversationalRetrievalChain`, a LangChain vector store, or a built-in
retriever abstraction.

| Component | Location | Replaces | Why it improves the design |
| --- | --- | --- | --- |
| `PromptTemplate` | `prompts/templates.py` | Manual string formatting | Makes the existing fixed wording runnable-compatible and centrally represented without changing any instruction text. |
| `RunnableLambda` | `llm/runnables.py`, `pipelines/rag_pipeline.py` | Direct glue calls between stages | Adapts project contracts to LCEL while keeping the custom retriever and `generate` methods intact. |
| `RunnablePassthrough.assign` | `pipelines/rag_pipeline.py` | Local temporary pipeline variables | Builds an inspectable state containing retrieval output, context, prompt, generation, and response. |
| `StrOutputParser` | `pipelines/rag_pipeline.py` | Direct final-text pass-through | Makes the text-output step explicit in the LCEL sequence. |

The custom components remain custom for specific reasons:

- **FAISS and retrieval strategy:** `Retriever` defines index validation,
  candidate count, chunk schema, metrics, and source attribution.
- **Cross-encoder reranker:** it works on the existing candidate dictionaries
  and adds `rerank_score` without converting the project to another document
  representation.
- **Ingestion and chunk handling:** the configured active splitter and the
  standalone paragraph-aware utility encode project-specific chunk semantics.
- **Prompt design:** only rendering changed; the wording remains owned by the
  project.
- **Offline and Groq backends:** custom adapters preserve LoRA discovery,
  memory controls, API endpoint details, system message, and metrics.

`scripts/ingest.py` already imported `langchain_text_splitters` before LCEL was
added. That pre-existing use is retained because changing the active chunking
implementation would alter index contents and is outside the orchestration
refactor.

## 10. Configuration

The active configuration file is `config/config.json`. Paths are interpreted
relative to the repository root when a caller resolves them. `config_old.json`
is not read by current runtime code; it is a legacy reference and should not
be treated as active configuration.

### Active settings

| Key | Current value | Controls | Consumers |
| --- | --- | --- | --- |
| `model.base_model` | `Qwen/Qwen2.5-3B-Instruct` | Base local causal language model. | `OfflineLLM`, `fine_tune.py`, offline UI label. |
| `model.adapter_repo_id` | `rahulboby/chatveritas-lora-adapter` | Hugging Face fallback for an incomplete local adapter; optional training upload target. | `OfflineLLM`, `fine_tune.py`, offline UI label. |
| `model.adapter_path` | `models/adapters` | Local output/input directory for the PEFT adapter. | `OfflineLLM`, `fine_tune.py`. |
| `embedding.model` | `sentence-transformers/all-MiniLM-L6-v2` | Document and query embedding model. | `ingest.py`, `RAGPipeline`/`Retriever`. |
| `embedding.device` | `cpu` | Device passed to SentenceTransformer. | `ingest.py`, `Retriever`. |
| `reranker.model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker. | `RAGPipeline`/`Retriever`. |
| `reranker.device` | `cpu` | Device passed to CrossEncoder. | `RAGPipeline`/`Reranker`. |
| `retrieval.top_k` | `5` | Number of reranked chunks returned and used as context. | `Retriever`. |
| `retrieval.chunk_size` | `1000` | Character chunk size during active ingestion. | `ingest.py`. |
| `retrieval.chunk_overlap` | `80` | Character overlap during active ingestion. | `ingest.py`. |
| `retrieval.faiss_candidates` | `30` | Maximum FAISS candidates passed to the reranker. | `Retriever`. |
| `generation.max_new_tokens` | `2048` | Local generation limit and Groq `max_tokens`. | Both LLM backends. |
| `generation.temperature` | `0.3` | Enables/controls sampling for local and Groq generation. | Both LLM backends. |
| `paths.raw_data` | `data/raw` | Source corpus directory. | `ingest.py`. |
| `paths.vectorstore` | `data/vectorstore` | FAISS index and chunk metadata directory. | Ingestion, retriever, dataset preparation, chunk printing. |
| `paths.processed_data` | `data/processed/train.jsonl` | Generated conversational training dataset. | Dataset preparation, training. |
| `paths.training_checkpoints` | `models/checkpoints` | Training checkpoint directory. | `fine_tune.py`. |
| `dataset.max_chunk_tokens` | `500` | Present in active config but not read by the current dataset-preparation script. | Reserved/no current consumer. |
| `llm.provider` | `groq` | Allowed provider for synthetic question generation. | `prepare_finetune_dataset.py`. |
| `llm.model` | `openai/gpt-oss-120b` | Groq model for synthetic question generation. | `QuestionGenerator`. |
| `llm.temperature` | `0.7` | Sampling for synthetic question generation. | `QuestionGenerator`. |
| `llm.max_retries` | `3` | Maximum question-generation attempts. | `QuestionGenerator`. |
| `cache.enabled` | `true` | Enables on-disk synthetic-response cache. | Dataset preparation. |
| `cache.directory` | `data/cache` | Cache location. | `CacheManager`. |
| `training.epochs` | `3` | Number of SFT epochs. | `fine_tune.py`. |
| `training.batch_size` | `2` | Per-device training batch size. | `fine_tune.py`. |
| `training.gradient_accumulation` | `8` | Gradient accumulation steps. | `fine_tune.py`. |
| `training.learning_rate` | `0.0002` | Optimizer learning rate. | `fine_tune.py`. |
| `training.max_sequence_length` | `2048` | SFT maximum sequence length. | `fine_tune.py`. |
| `lora.r` | `16` | Low-rank adapter rank. | `fine_tune.py`. |
| `lora.alpha` | `32` | LoRA scaling factor. | `fine_tune.py`. |
| `lora.dropout` | `0.05` | LoRA dropout. | `fine_tune.py`. |
| `lora.target_modules` | Qwen projection module list | Modules adapted by LoRA. | `fine_tune.py`. |

### Optional keys accepted by code

| Key | Code default | Effect |
| --- | --- | --- |
| `generation.model` | `openai/gpt-oss-120b` | Groq deployment model. This is distinct from `llm.model`, which is for dataset preparation. |
| `inference.max_cpu_memory` | `12GiB` | Offline model CPU memory budget. |
| `inference.max_gpu_memory` | `4GiB` | Offline model GPU memory budget when CUDA is available. |
| `inference.offload_directory` | `data/model_offload` | Folder for Hugging Face model offload files. |
| `training.seed` | `42` | Grouped train/validation split seed and SFT data seed. |
| `training.validation_split` | `0.05` | Fraction of source-chunk groups held out for validation. |
| `training.eval_batch_size` | `1` | Per-device evaluation batch size. |
| `training.gradient_checkpointing` | `true` | Enables k-bit model preparation and trainer gradient checkpointing. |
| `training.packing` | `false` | Enables/disables SFT sequence packing. |
| `training.optimizer` | `paged_adamw_8bit` | Trainer optimizer identifier. |
| `training.lr_scheduler` | `cosine` | Learning-rate schedule. |
| `training.warmup_ratio` | `0.03` | Trainer warmup proportion. |
| `training.weight_decay` | `0.0` | Trainer weight decay. |
| `training.max_grad_norm` | `1.0` | Gradient clipping norm. |
| `training.logging_steps` | `5` | Trainer logging interval. |
| `training.save_steps` | `50` | Checkpoint interval. |
| `training.eval_steps` | `50` | Validation interval when validation data exists. |
| `training.save_total_limit` | `2` | Number of checkpoints retained. |
| `lora.bias` | `none` | LoRA bias training policy. |
| `hub.private` | `false` | Visibility used when creating the optional adapter repository. |

## 11. External Dependencies

| Dependency | Purpose in this repository | Used in |
| --- | --- | --- |
| PyTorch | Local tensor operations, inference mode, device/memory checks, CUDA cache handling. | `llm/offline.py`, training. |
| Transformers | Qwen tokenizer/model loading, chat template, generation, tokenizer-backed utility chunking. | Offline LLM, paragraph utility, training. |
| PEFT | Loads the inference LoRA adapter and defines QLoRA training configuration. | Offline LLM, training. |
| Sentence Transformers | Sentence embedding model and CrossEncoder reranker model. | Ingestion and retrieval. |
| FAISS | Stores and exhaustively searches float32 vectors with L2 distance. | Ingestion and retrieval. |
| NumPy | Ensures embedding/query arrays are `float32` and calculates average distances. | Ingestion and retrieval. |
| LangChain Core | `PromptTemplate`, LCEL runnables, passthrough assignment, and text output parser. | Prompts, LLM adapter, pipelines. |
| LangChain Text Splitters | Configured character splitter used by active ingestion. | `scripts/ingest.py`. |
| Streamlit | Chat UI, resource/data caching, expanders, metrics, and session history. | Both app entry points. |
| OpenAI client | OpenAI-compatible client configured with Groq base URL. | Deploy generation. |
| Groq SDK | Synthetic question-generation client. | `QuestionGenerator`. |
| TRL, Datasets, BitsAndBytes, Accelerate | Dataset construction, 4-bit QLoRA training, supervised fine-tuning, placement support. | `scripts/fine_tune.py`. |
| Hugging Face Hub | Optional adapter-repository creation and upload. | `scripts/fine_tune.py`. |
| Trafilatura | Optional URL extraction into source text files. | `scripts/scrape_urls.py`. |
| python-dotenv | Loads `GROQ_API_KEY` from `.env`. | Deploy app and dataset preparation. |

## 12. Engineering Decisions

### FAISS `IndexFlatL2`

The code uses a flat L2 index, which provides exhaustive nearest-neighbor
search over the current vector store and keeps index behavior straightforward.
The trade-off is memory and query cost at much larger corpus sizes; the code
does not claim an approximate index or introduce compression.

### Cross-encoder reranking

FAISS provides vector similarity candidates, while the CrossEncoder scores the
question and full chunk jointly. The system performs the more expensive model
scoring only for `faiss_candidates`, then keeps `top_k`. This preserves a
distinct candidate-recall step and final relevance-ordering step.

### LoRA / QLoRA

The project trains a small PEFT adapter rather than full model weights. That
keeps the base model external and makes the trained artifact portable. Training
uses 4-bit loading and k-bit preparation to reduce GPU memory requirements.

### Chunking and corpus alignment

The active ingestion splitter prioritizes paragraph/newline/sentence
boundaries while operating with character limits. Dataset generation then reads
the saved chunks rather than recreating them. This prevents the fine-tuning
answer text from drifting away from the retrieval corpus.

### Streamlit plus terminal entry points

Streamlit provides a practical local web interface with cached heavyweight
resources and evidence inspection. The terminal script provides a minimal
interactive option without UI state. Both use `ChatVeritas`, so they do not
fork application behavior.

### Groq deployment backend

Hosted generation makes deployment feasible where a Qwen 3B model and adapter
cannot be hosted within available resources. Retrieval remains local/project
controlled, so changing the LLM backend does not change source selection.

### LangChain Core, not a prebuilt RAG chain

LCEL improves composition, traceability, and separation of pipeline state.
Prebuilt retrieval chains would hide the index, reranking, prompt, and metrics
contracts that are the central engineering decisions of this application.

### Modular application architecture

The directory boundaries map to responsibilities rather than framework types.
They permit independent evolution of the retriever, prompt, model backend, and
entry points while preserving one public `ChatVeritas` interface.

## 13. Project Lifecycle

The following is the full developer workflow from an empty local corpus.

1. **Install the environment.** Install `requirements.txt`; configure a
   `GROQ_API_KEY` for dataset generation and deploy mode.
2. **Prepare the corpus.** Manually clean content and store `.txt` files in
   `data/raw/`. Optionally edit `scripts/scrape_urls.py` and run it to create
   source files from selected URLs.
3. **Build retrieval artifacts.** Run `python scripts/ingest.py`. Confirm
   `index.faiss` and `chunks.pkl` are generated in `data/vectorstore/`.
4. **Inspect the corpus if needed.** Run `python scripts/print_chunks.py` to
   check source attribution, identifiers, and chunk text.
5. **Create training records.** Run
   `python scripts/prepare_finetune_dataset.py`. This requires the vector store
   and Groq credentials and produces `data/processed/train.jsonl`.
6. **Validate before training.** Run
   `python scripts/fine_tune.py --validate-only`. Correct data/configuration
   failures before allocating GPU resources.
7. **Train or resume.** Run `python scripts/fine_tune.py`, or add `--resume`
   to use the latest checkpoint. Expect checkpoints and a completed adapter.
8. **Run inference.** Use `app_offline.py` or terminal chat for local Qwen,
   optionally with the adapter; use `app_deploy.py` for Groq generation.
9. **Iterate safely.** Any embedding-model or chunking change requires a new
   vector store. Any base-model change requires a compatible newly trained
   adapter. Prompt changes require no artifact regeneration but affect
   generation behavior.

## 14. Future Roadmap

No explicit future-version roadmap, TODO list, issue reference, or planned
feature list exists in the current source documentation. Accordingly, this
document does not assert plans for hybrid retrieval, BM25, reciprocal-rank
fusion, conversation memory, evaluation tooling, semantic chunking, or
benchmarking. Those would be new proposals rather than documented repository
plans.

## 15. Developer Notes

### Common extension points

- **Prompt changes:** Edit only `prompts/templates.py`. Preserve `{context}`
  and `{question}` unless `RAGPipeline._prompt_inputs` is changed in tandem.
- **New LLM backend:** Implement `BaseLLM.generate(prompt)`, return the same
  `(response, metrics)` tuple, and add a thin `RAGPipeline` specialization.
  `as_runnable()` then integrates it with LCEL automatically.
- **Embedding model:** Change `embedding.model` and rerun ingestion before
  using runtime retrieval. Do not mix an index built with one model and queries
  embedded by another.
- **Reranker:** Change `reranker.model` or device for a compatible
  CrossEncoder. To alter ranking behavior, update `Reranker.rerank` while
  retaining the candidate dictionary contract expected by the UI.
- **Retrieval policy:** Adjust `top_k` or `faiss_candidates` first. Changes to
  the index type or chunk schema require coordinated changes in ingestion and
  retrieval.
- **Training:** Keep LoRA target modules compatible with the selected base
  architecture. Use a new adapter directory when changing base models.

### Debugging and observability

- Start with the Streamlit **RAG Metrics** and **Retrieved Context** expanders.
  They expose retrieved chunks, FAISS distance, cross-encoder score, sources,
  and timing metrics.
- Logs are emitted to stdout in a consistent module-qualified format. Use the
  logger names to isolate retrieval, pipeline, or backend failures.
- A missing or out-of-sync vector store fails during `Retriever` construction;
  rerun ingestion instead of bypassing its count validation.
- Missing `GROQ_API_KEY` fails early in deploy generation and dataset
  preparation. Verify `.env` loading and environment scope.
- For local loading problems, inspect adapter completeness and the configured
  base model; `OfflineLLM` logs whether it chose a local adapter, Hub fallback,
  or base-model fallback.

### Performance profiling

The result metrics are the first profiling surface:

- `embedding_time_ms` identifies query-embedding cost.
- `retrieval_time_ms` identifies FAISS search cost.
- `reranking_time_ms` identifies CrossEncoder cost.
- `generation_time` identifies LLM latency.
- `prompt_tokens`, `faiss_candidates`, and `retrieved_chunks` explain the
  amount of work at each stage.

For controlled comparisons, change one configuration value at a time, retain
the same corpus/index, and record the returned metric dictionary alongside
answer quality observations.

"""
)