# Streamlit Pages and Model-Loading Flow

This document explains what each Streamlit page does, what happens before a
generation model is loaded, and how the page connects to the ChatVeritas RAG
pipeline.

## Application entry point

Run the application with:

```powershell
streamlit run app.py
```

`app.py` is the home page and the multipage entry point. Streamlit discovers
the numbered scripts in `pages/` and presents them in the sidebar in numeric
order.

The entry page:

1. Sets the application title, icon, and wide layout.
2. Explains the three available pages and their generation backends.
3. Tells the user that `.txt` files must be ingested before chat can use them.

It does not import `ChatVeritas`, construct a pipeline, or load any embedding,
reranker, or generation model.

## Shared initialization model

The two chat pages use the same high-level sequence:

```text
Streamlit page rerun
  -> imports and safe runtime setup
  -> load environment variables and settings
  -> render backend controls
  -> wait for an explicit Load button click
  -> set a session-state load flag and rerun
  -> construct ChatVeritas
  -> construct the shared retriever
  -> load embedding model, FAISS index, and cross-encoder
  -> construct the selected generation backend
  -> cache the initialized ChatVeritas instance
  -> accept chat questions
```

The phrase "load the model" includes several different resources:

- **Embedding model:** converts a question into a vector for FAISS search.
- **FAISS index and chunks:** load the already-generated local vector store.
- **Cross-encoder:** reranks the FAISS candidates.
- **Generation model/client:** produces the final answer. This is a local
  Hugging Face model, an LM Studio client, or a Groq client depending on the
  page and selection.

`ChatVeritas` creates a pipeline in its constructor. `RAGPipeline` creates the
retriever first and then the generation backend. Therefore, chat initialization
can fail before generation is available if the vector store is missing or
invalid.

Both chat pages decorate their loader with `st.cache_resource`. Once a
particular backend configuration has loaded successfully, later Streamlit
reruns reuse that resource instead of loading it again. The Offline page's
cache key includes the backend and LoRA choice; the Cloud page has one cached
resource for its Groq pipeline.

## Page 1: Offline Chat

File: `pages/1_Offline_Chat.py`

### Purpose

Provides document-grounded chat using either:

- **Local Model:** the configured Hugging Face Qwen model, optionally combined
  with a PEFT LoRA adapter.
- **LM Studio (Local):** an OpenAI-compatible local endpoint. The current
  defaults point to the configured local endpoint in `configs/settings.py`.

Retrieval remains local for both choices.

### What runs before model loading

On each page execution, the script:

1. Adds the project root to `sys.path` so package imports work when Streamlit
   executes the page script.
2. Applies thread limits from `core.constants`.
3. Imports Streamlit, settings, the `ChatVeritas` facade, and logging helpers.
4. Enables `faulthandler` when the runtime permits it.
5. Loads variables from the project `.env` file.
6. Sets the page configuration and renders the backend settings sidebar.
7. Lets the user choose the generation backend. The LoRA checkbox is enabled
   only for the local Hugging Face option.
8. Initializes `offline_load_requested` and `offline_loaded` in
   `st.session_state` if they do not exist.

None of these steps constructs `ChatVeritas` or loads a machine-learning
model. The settings captions only display configured identifiers and URLs.

### Explicit load sequence

When the user clicks **Load Chat**, the page sets
`offline_load_requested = True`, resets `offline_loaded`, and calls
`st.rerun()`. On the rerun, the page calls:

```python
load_chatbot(backend, use_lora)
```

The cached function constructs `ChatVeritas(mode="offline", ...)`, which
creates an `OfflinePipeline`:

1. `Retriever` loads `index.faiss` and `chunks.pkl`.
2. `SentenceTransformer` loads the embedding model.
3. `CrossEncoder` loads the reranker.
4. `OfflineLLM` selects the generation path.

For **Local Model**, `OfflineLLM` then:

1. Resolves a local adapter in `models/adapters/`, or falls back to the
   configured Hugging Face adapter repository when the local adapter is
   incomplete.
2. Loads the tokenizer.
3. Loads the base Qwen model with automatic device mapping and configured CPU
   or GPU memory limits.
4. Applies the LoRA adapter when enabled. If adapter loading fails, it logs a
   warning and continues with the base model.
5. Switches the model to evaluation mode.

For **LM Studio (Local)**, no Hugging Face generation weights are loaded by
ChatVeritas. `OfflineLLM` validates the endpoint/model settings and creates an
OpenAI-compatible client. The separate LM Studio/Ollama server owns the
generation model.

### After loading

The page initializes `offline_messages`, redraws prior messages, and enables
`st.chat_input`. A question calls `chatbot.ask(prompt)`, which runs retrieval,
context assembly, prompt construction, generation, and result formatting.
The response, RAG metrics, and retrieved evidence chunks are displayed.

## Page 2: Cloud Chat

File: `pages/2_Cloud_Chat.py`

### Purpose

Provides the same grounded retrieval experience while sending generation to
Groq. The question embedding, FAISS search, chunk loading, and cross-encoder
reranking remain local.

### What runs before model loading

The page:

1. Adds the project root to `sys.path` and applies thread limits.
2. Imports Streamlit, settings, `ChatVeritas`, logging, and dotenv support.
3. Enables `faulthandler` when possible and loads `.env`.
4. Sets page configuration and displays the Groq provider/model information.
5. Initializes `cloud_load_requested` and `cloud_loaded` in session state.

The Groq API key is not needed while rendering the page. It is resolved when
the page actually constructs the cloud pipeline.

### Explicit load sequence

Clicking **Load Cloud Chat** sets the request flag and triggers a rerun. The
rerun calls the cached `load_chatbot()` function, which constructs
`ChatVeritas(mode="deploy")` and therefore a `DeployPipeline`:

1. The shared `Retriever` loads the local FAISS index, chunks, embedding model,
   and cross-encoder.
2. `DeployLLM` resolves the Groq provider, validates `GROQ_API_KEY`, and creates
   an OpenAI-compatible client configured with the Groq base URL and model ID.

No Groq model weights are downloaded into this application. The remote Groq
service owns and runs the generation model. The first network generation
request happens only after the user submits a question.

### After loading

The page initializes the separate `cloud_messages` history, redraws prior
messages, and enables chat input. Each question follows the shared RAG flow and
displays the answer, metrics, and retrieved context. Retrieval errors and
generation/API errors are shown with technical traceback details in an
expander.

## Page 3: Architecture

File: `pages/3_Architecture.py`

### Purpose

Displays `ChatVeritas_ARCHITECTURE.md` inside Streamlit. Normal Markdown is
rendered with `st.markdown`; Mermaid code blocks are extracted with a regular
expression and rendered with `streamlit_mermaid`.

### What happens before model loading

There is no model-loading stage on this page. It:

1. Adds the project root to `sys.path`.
2. Imports Streamlit and the Mermaid component.
3. Sets page configuration.
4. Reads the architecture Markdown file from disk.
5. Renders text and Mermaid diagrams.

It does not import `ChatVeritas`, create a retriever, contact Groq/LM Studio,
or load a local generation model. If the Markdown file cannot be read, the
page displays an error instead.

## Failure and stop behavior

Both chat pages stop before chat input when initialization has not been
requested or when loading fails. On failure they:

- clear the relevant load flags;
- display a user-facing error;
- expose the Python traceback in a technical-details expander; and
- call `st.stop()` so the page cannot submit questions with an uninitialized
  chatbot.

Typical pre-generation failures are missing or inconsistent vector-store
files, unavailable embedding/reranker dependencies, invalid provider settings,
missing `GROQ_API_KEY`, an unavailable local endpoint, or a failed local model
or adapter load.

## Source map

| Concern | Implementation |
|---|---|
| Home and page discovery | `app.py` |
| Offline page controls and lifecycle | `pages/1_Offline_Chat.py` |
| Cloud page controls and lifecycle | `pages/2_Cloud_Chat.py` |
| Architecture rendering | `pages/3_Architecture.py` |
| Facade and mode selection | `core/chatveritas.py` |
| Retriever construction and search | `core/retrieval.py` |
| RAG orchestration | `core/pipelines.py` |
| Generation backends | `core/llm.py` |
| Runtime settings | `configs/settings.py` |