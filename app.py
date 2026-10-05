"""Purpose: Run the Streamlit RAG chatbot with retrieval, reranking, and streamed answers.
Dependencies: built-in: os, sys, time, textwrap, traceback, pathlib, faulthandler; installed: faiss, streamlit, openai, python-dotenv.
Custom: utils.config_loader, utils.vectorstores, utils.retriever.
"""
import os
import sys
import time
import textwrap
import traceback
from pathlib import Path

# ========== THREADING LIMITS ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ---- Set project root and adjust sys.path BEFORE importing project modules ----
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# ---- Third-party imports ----
import faulthandler
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ---- Project imports ----
from utils.config_loader import load_config
from utils.chat_history import (
    build_request_messages,
    parse_chat_command,
    record_completed_turn,
)
from utils.vectorstores import list_vectorstores

load_dotenv()
faulthandler.enable(all_threads=True)
st.set_page_config(page_title="ChatVeritas", layout="wide", page_icon="💬")

# ---------- Cache config loader ----------
@st.cache_data
def get_config():
    """Load and cache the configuration."""
    return load_config()

config = get_config()

def create_client(config, provider):
    if provider == "ollama":
        active_llm = config["model_ollama"]
        api_key = "ollama"
    else:
        active_llm = config["llm"]
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY to use the Groq endpoint.")

    return OpenAI(api_key=api_key, base_url=active_llm["url"])

# ---------- Load components with checkpoints ----------
@st.cache_resource
def load_components(config, vectorstore_path):
    from utils.retriever import Retriever

    retriever = Retriever(
        index_path=Path(vectorstore_path) / "index.faiss",
        chunks_path=Path(vectorstore_path) / "chunks.pkl",
        embedding_model=config["embedding"]["model"],
        top_k=config["retrieval"]["top_k"],
        faiss_candidates=config["retrieval"]["faiss_candidates"],
        embedding_device=config["embedding"].get("device", "cpu"),
        reranker_model=config["reranker"]["model"],
        reranker_device=config["reranker"].get("device", "cpu"),
    )
    return retriever

# ---------- Streaming generator ----------
def generate_response_stream(question, retriever, config, client, history=None):
    """
    Generator that yields tokens from the API while also building the full response.
    After streaming completes, it stores the final response, chunks, and metrics
    in st.session_state['_stream_result'].
    """
    # ---- Retrieval ----
    retrieval = retriever.retrieve(question)
    chunks = retrieval["results"]
    metrics = retrieval["metrics"]

    # Build context
    context = "\n\n".join(item["chunk"] for item in chunks)

    # ---- Build prompt ----
    prompt = textwrap.dedent(f"""
        You are an expert technical assistant answering questions about the provided documents.
        Use the retrieved context as your PRIMARY source of information.
        Guidelines:
        1. Base your answer primarily on the provided context.
        2. If the answer is explicitly stated in the context, answer confidently.
        3. If the answer is not explicitly stated but can be reasonably inferred, clearly state it is an inference.
        4. Only respond with "I don't have enough information in the provided documents." if the context is insufficient.
        5. Never invent facts.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """).strip()

    gen_start = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=config["active_llm"]["model"],
            messages=build_request_messages(
                system_prompt=(
                    "You are ChatVeritas, a document-grounded AI assistant. "
                    "Answer only using the supplied context. "
                    "If the answer is not present, clearly state that there "
                    "is insufficient information."
                ),
                current_prompt=prompt,
                provider=config["active_llm"]["provider"],
                history=history,
            ),
            temperature=config["active_llm"].get(
                "temperature",
                config["generation"]["temperature"],
            ),
            max_tokens=config["generation"]["max_new_tokens"],
            stream=True,   # <-- enable streaming
        )
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}")

    full_response = ""
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token
            yield token
        # Capture usage from the final chunk (if available)
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            total_tokens = chunk.usage.total_tokens

    record_completed_turn(
        history if history is not None else [],
        config["active_llm"]["provider"],
        question,
        full_response,
    )

    # If usage wasn't provided in the stream, we can try to get it from the last chunk
    # but it's usually included.
    metrics["generation_time"] = time.perf_counter() - gen_start
    metrics["prompt_tokens"] = prompt_tokens or 0

    # Store final data in session state for later display
    st.session_state["_stream_result"] = {
        "response": full_response.strip(),
        "chunks": chunks,
        "metrics": metrics,
    }

    # Optionally yield a final sentinel (not needed for st.write_stream)
    # but we can yield an empty string to finish.

# ---------- STREAMLIT UI ----------
st.title("ChatVeritas: Two-Stage RAG Chatbot (FAISS + Cross-Encoder)")

st.info(
    """
    **System Architecture**

    ChatVeritas features a **Two-Stage RAG Pipeline**:
    1. **Retrieval**: FAISS dense vector search over document chunk embeddings.
    2. **Reranking**: Cross-Encoder model to score and reorder top candidates for context accuracy.
    3. **Generation**: LLM inference grounded strictly on the retrieved context.
    """
)

config = get_config()
vectorstores = list_vectorstores(PROJECT_ROOT, config)
if not vectorstores:
    st.error("No vector stores are available. Run scripts/ingest.py to create one.")
    st.stop()

vectorstore_names = [path.name for path in vectorstores]
selected_name = st.sidebar.selectbox("Vector store", vectorstore_names)
selected_vectorstore = next(path for path in vectorstores if path.name == selected_name)
selected_provider = st.sidebar.selectbox(
    "Response endpoint",
    ("Ollama", "Groq"),
).casefold()
active_llm = config["model_ollama"] if selected_provider == "ollama" else config["llm"]
active_config = {**config, "active_llm": active_llm}
if selected_provider == "ollama":
    st.sidebar.caption("Ollama retains question-and-answer history; retrieved context is not retained.")
else:
    st.sidebar.caption("Groq requests are stateless; prior turns are not sent.")

if "history_provider" not in st.session_state:
    st.session_state.history_provider = selected_provider
    st.session_state.ollama_history = []
elif st.session_state.history_provider != selected_provider:
    st.session_state.history_provider = selected_provider
    st.session_state.ollama_history = []

# Load the selected store and API client only after the user has chosen a store.
try:
    client = create_client(config, selected_provider)
    retriever = load_components(config, str(selected_vectorstore))
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.code(traceback.format_exc(), language="python")
    st.stop()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ollama_history" not in st.session_state:
    st.session_state.ollama_history = []

notice = st.session_state.pop("chat_notice", None)
if notice:
    st.info(notice)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    command = parse_chat_command(prompt)
    if command == "/clear":
        st.session_state.messages = []
        st.session_state.ollama_history = []
        st.session_state.chat_notice = "Conversation history cleared."
        st.rerun()
    if command in {"/bye", "/exit"}:
        st.session_state.chat_notice = (
            "Exit commands work only in the terminal CLI; this web chat remains open."
        )
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Use st.write_stream to display the generator output in real time
            stream_gen = generate_response_stream(
                prompt,
                retriever,
                active_config,
                client,
                st.session_state.ollama_history,
            )
            final_text = st.write_stream(stream_gen)  # returns the concatenated text
        except Exception as e:
            st.error(f"Error during generation: {e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc(), language="python")
            st.stop()

        # After streaming, retrieve the stored result
        result = st.session_state.pop("_stream_result", None)
        if result is not None:
            response = result["response"]
            chunks = result["chunks"]
            metrics = result["metrics"]
        else:
            # fallback (should not happen)
            response = final_text or "(No response)"
            chunks = []
            metrics = {}

        # ---- Metrics and context expanders ----
        with st.expander("RAG Metrics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding Time", f"{metrics.get('embedding_time_ms', 0.0):.2f} ms")
                st.metric("Retrieval Time", f"{metrics.get('retrieval_time_ms', 0.0):.2f} ms")
            with col2:
                st.metric("Re-ranking Time", f"{metrics.get('reranking_time_ms', 0.0):.2f} ms")
                st.metric("Generation Time", f"{metrics.get('generation_time', 0.0):.2f} s")
                st.metric("Prompt Tokens", metrics.get("prompt_tokens", 0))
            with col3:
                st.metric("Retrieved Chunks", metrics.get("retrieved_chunks", len(chunks)))
                st.metric("Avg L2 Distance", f"{metrics.get('average_distance', 0.0):.3f}")

        with st.expander("Retrieved Context"):
            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"### Chunk {i}")
                    st.markdown(
                        f"**Source:** {chunk.get('source', 'Unknown')}  \n"
                        f"**Chunk ID:** {chunk.get('chunk_id', 'N/A')}  \n"
                        f"**FAISS L2:** {chunk.get('distance', 0.0):.3f}  \n"
                        f"**Cross-Encoder:** {chunk.get('rerank_score', 0.0):.3f}"
                    )
                    st.write(chunk.get("chunk", ""))
            else:
                st.info("No relevant documents were retrieved.")

            st.markdown("### Sources Used")
            sources = metrics.get("sources", [])
            if sources:
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.write("No sources available.")

    st.session_state.messages.append({"role": "assistant", "content": response})