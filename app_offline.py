import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path must be extended before project imports so that Python can
# resolve application modules when this file is run directly or via Streamlit.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Thread limits MUST be applied before any NumPy / BLAS / FAISS /
# HuggingFace library is imported.  apply_thread_limits() only sets
# os.environ and has no heavy dependencies of its own.
# ---------------------------------------------------------------------------
from core.constants import apply_thread_limits
apply_thread_limits()

import faulthandler
import streamlit as st

from core.config import load_config
from core.logger import get_logger
from interfaces.chatveritas import ChatVeritas

faulthandler.enable(all_threads=True)

logger = get_logger(__name__)


# ---------- Cached configuration ----------
@st.cache_data
def get_config() -> dict:
    """Load and cache the application configuration."""
    return load_config()


# ---------- Cached chatbot (re-instantiated when use_lora changes) ----------
@st.cache_resource
def load_chatbot(use_lora: bool) -> ChatVeritas:
    """Instantiate and cache the offline ChatVeritas chatbot."""
    logger.info("Loading ChatVeritas offline chatbot (use_lora=%s).", use_lora)
    config = get_config()
    return ChatVeritas(mode="offline", use_lora=use_lora, config=config)


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="ChatVeritas – Local LLM + LoRA", layout="wide")
st.title("ChatVeritas: Fine-Tuned Two-Stage RAG Chatbot (Local)")

# ---------- Sidebar for LoRA toggle ----------
with st.sidebar:
    st.header("Model Settings")
    use_lora = st.checkbox(
        "Use LoRA adapter",
        value=True,
        help="Uncheck to use the base model without fine-tuning."
    )
    if use_lora:
        st.info("LoRA adapter will be loaded from the Hugging Face repo specified in `config.json`.")
    else:
        st.info("Using the base model only.")

    config = get_config()
    st.caption(f"Base model: `{config['model']['base_model']}`")
    st.caption(f"Adapter repo: `{config['model']['adapter_repo_id']}`")

# ---------- Load chatbot ----------
try:
    chatbot = load_chatbot(use_lora)
except Exception as e:
    st.error(f"Failed to load ChatVeritas: {e}")
    st.code(traceback.format_exc(), language="python")
    st.stop()

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- User input ----------
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chatbot.ask(prompt)
                response = result["response"]
                chunks = result["chunks"]
                metrics = result["metrics"]
            except Exception as e:
                logger.error("Error during generation: %s", e, exc_info=True)
                st.error(f"Error during generation: {e}")
                st.code(traceback.format_exc(), language="python")
                st.stop()

        # Display the assistant's answer
        st.markdown(response)

        # ---------- Expanders for debugging ----------
        with st.expander("RAG Metrics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding Time", f"{metrics.get('embedding_time_ms', 0.0):.2f} ms")
                st.metric("Retrieval Time", f"{metrics.get('retrieval_time_ms', 0.0):.2f} ms")
            with col2:
                st.metric("Re‑ranking Time", f"{metrics.get('reranking_time_ms', 0.0):.2f} ms")
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
                        f"**Cross‑Encoder:** {chunk.get('rerank_score', 0.0):.3f}"
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
