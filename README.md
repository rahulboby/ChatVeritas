<div align="center">

# ChatVeritas

### Document-grounded answers, with a direct-chat mode when you do not need retrieval.

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FAISS-Vector%20search-0467DF" alt="FAISS vector search">
  <img src="https://img.shields.io/badge/Sentence--Transformers-Embeddings-32A852" alt="Sentence Transformers embeddings">
  <img src="https://img.shields.io/badge/Groq%20%7C%20Ollama-OpenAI--compatible%20APIs-6E56CF" alt="Groq and Ollama APIs">
</p>

</div>

ChatVeritas is a Python chat application for exploring information in your own documents. It combines semantic retrieval with cross-encoder reranking to provide answers grounded in a selected document collection. For questions that do not need a knowledge base, **General chatting** sends the conversation directly to the configured language-model API without retrieval.

## At a glance

| Mode | What happens |
| --- | --- |
| **Document chat (RAG)** | A question is embedded, matching document chunks are retrieved from FAISS and reranked, then supplied to the language model as context. |
| **General chatting** | The question goes directly to the language model. No vector store, embeddings, or retrieval are used. |

The Streamlit app also accepts temporary PDF and TXT uploads for document chat. The terminal interface supports both general chat and saved vector stores.

## How it works

### Build a knowledge base

```text
TXT files in data/raw/
        │
        ▼
Chunk documents → Sentence Transformer embeddings → FAISS index
                                                   + chunk metadata
```

Run `python scripts/ingest.py` to create a named vector store from the `.txt` files in `data/raw/`. The index and corresponding chunk metadata are saved together so retrieved passages can be shown with their source.

### Answer a question

```text
Question → embed → FAISS candidate search → Cross-Encoder rerank
                                             │
                                             ▼
                                  Context + question → Chat API → Answer
```

The configured embedding model is `sentence-transformers/all-MiniLM-L6-v2`; FAISS performs L2 search and a Cross-Encoder scores the candidate passages before they are sent to the language model. In **General chatting**, the flow skips the retrieval stages and sends the request directly to the API.

## Run

```bash
streamlit run app.py       # Web interface
python scripts/chat.py     # Terminal interface
```

Choose **General chatting** to start without a vector store, or select a saved store for RAG. In the web app, temporary PDF/TXT uploads can be used instead of a saved store.

The configured endpoints are Groq and Ollama. Set `GROQ_API_KEY` to use Groq; configure and run Ollama locally to use that endpoint. Model names, endpoints, retrieval settings, and local model settings are in [`config/config.json`](config/config.json).

## Technology

- **Interface:** Streamlit web app and terminal chat
- **Ingestion:** text cleaning, recursive chunking, Sentence Transformers
- **Retrieval:** FAISS vector search followed by Cross-Encoder reranking
- **Generation:** OpenAI-compatible chat completions API (Groq or Ollama)
- **Document inputs:** `.txt` files for saved stores; temporary `.txt` and PDF uploads in the web app

## Project map

```text
app.py                  Streamlit interface and temporary-upload workflow
scripts/chat.py         Terminal chat interface
scripts/ingest.py       Build a named FAISS vector store from data/raw/*.txt
scripts/convert_pdfs.py Parse uploads and build a temporary retrieval index
utils/retriever.py      Embed questions, search FAISS, and rerank results
config/config.json      API, model, retrieval, and path configuration
```

## Roadmap

- [x] Direct general chat alongside document-grounded RAG
- [x] Two-stage retrieval with FAISS search and Cross-Encoder reranking
- [x] Saved vector stores and temporary PDF/TXT document chat
- [ ] Add a repeatable evaluation set for retrieval relevance and answer grounding
- [ ] Improve answer attribution with clearer source references

The roadmap reflects current project direction; future items may change.
