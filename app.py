"""
app_deploy.py - ChatVeritas: Fine-Tuned Two-Stage RAG Chatbot

This Streamlit app implements a document-grounded question-answering system.
It retrieves relevant chunks from a FAISS index, reranks them with a cross-encoder,
and generates an answer using the API (with streaming support).

The retrieval pipeline uses a fine-tuned embedding model and a reranker.
The generation step streams output token by token for a responsive UX.
"""
import os
import sys
import time
import textwrap
import traceback
from pathlib import Path

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
from utils.retriever import Retriever

# ========== THREADING LIMITS (prevent tqdm & BLAS threads from crashing) ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE"] = "1"          # kill tqdm monitor thread
os.environ["TRANSFORMERS_VERBOSITY"] = "error"   # suppress HF warnings

load_dotenv()
if not os.getenv("API_KEY") and not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Neither API_KEY nor GROQ_API_KEY found in environment variables.")

faulthandler.enable(all_threads=True)

# ---------- Cache config loader ----------
@st.cache_data
def get_config():
    """Load and cache the configuration."""
    return load_config()

config = get_config()
client = OpenAI(
    api_key=os.getenv("API_KEY") or os.getenv("GROQ_API_KEY"),
    base_url=config["llm"].get("url", "https://api.groq.com/openai/v1")
)
provider = config["llm"].get("provider", "no-provider-specified")

# ---------- Load components with checkpoints ----------
@st.cache_resource
def load_components(config):
    retriever = Retriever(
        index_path=PROJECT_ROOT / config["paths"]["vectorstore"] / "index.faiss",
        chunks_path=PROJECT_ROOT / config["paths"]["vectorstore"] / "chunks.pkl",
        embedding_model=config["embedding"]["model"],
        top_k=config["retrieval"]["top_k"],
        faiss_candidates=config["retrieval"]["faiss_candidates"],
        embedding_device=config["embedding"].get("device", "cpu"),
        reranker_model=config["reranker"]["model"],
        reranker_device=config["reranker"].get("device", "cpu"),
    )
    return retriever

# ---------- Streaming generator ----------
def generate_response_stream(question, retriever, config):
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
            model=config["llm"].get("model", "no-model-specified"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ChatVeritas, a document-grounded AI assistant. "
                        "Answer only using the supplied context. "
                        "If the answer is not present, clearly state that there "
                        "is insufficient information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=config["generation"]["temperature"],
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
st.set_page_config(page_title="ChatVeritas", layout="wide", page_icon="💬")
st.title("ChatVeritas: Fine-Tuned Two-Stage RAG Chatbot on Custom Dataset")

st.info(
    """
    **Deployment Notice**

    This public deployment uses the **GROQ API** for language model inference.

    The original ChatVeritas research system includes a fine-tuned LoRA model.
    That model is not included here because its size exceeds the limits of free
    cloud deployment platforms.

    The complete retrieval pipeline—including FAISS retrieval, reranking,
    and context-grounded generation—remains unchanged.
    """
)

# Load config and components
try:
    config = get_config()
    retriever = load_components(config)
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.code(traceback.format_exc(), language="python")
    st.stop()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Use st.write_stream to display the generator output in real time
            stream_gen = generate_response_stream(prompt, retriever, config)
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