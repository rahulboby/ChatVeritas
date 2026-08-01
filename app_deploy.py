import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path must be extended before any project imports.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Thread limits before any heavy library imports.
# ---------------------------------------------------------------------------
from core.constants import apply_thread_limits
apply_thread_limits()

import faulthandler
import streamlit as st
from dotenv import load_dotenv

from core.config import load_config
from core.logger import get_logger
from interfaces.chatveritas import ChatVeritas

faulthandler.enable(all_threads=True)
load_dotenv()

logger = get_logger(__name__)


# ---------- Cached configuration ----------
@st.cache_data
def get_config() -> dict:
    """Load and cache the application configuration."""
    return load_config()


# ---------- Cached chatbot ----------
@st.cache_resource
def load_chatbot() -> ChatVeritas:
    """Instantiate and cache the deploy ChatVeritas chatbot."""
    logger.info("Loading ChatVeritas deploy chatbot.")
    config = get_config()
    return ChatVeritas(mode="deploy", config=config)


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="ChatVeritas", layout="wide", page_icon="💬")
st.title("ChatVeritas: Fine-Tuned Two-Stage RAG Chatbot on Custom Dataset")

st.info(
    """
    **Deployment Notice**

    This public deployment uses the **Groq API** for language model inference.

    The original ChatVeritas research system includes a Qwen2.5 (3B) model fine tuned with LoRA.
    That model is not included here because its size exceeds the limits of free cloud deployment platforms.

    The complete retrieval pipeline—including FAISS retrieval, reranking, and context-grounded generation—remains unchanged.

    ChatVeritas is currently fine-tuned on the research paper and project report of dataveritas. Any questions about DataVeritas would be answered efficiently. The model would not answer anything other than the scope of the data given to it.
    """
)

st.link_button(
    "View the ChatVeritas Architecture",
    url = "architecture-chatveritas.streamlit.app"
)

# ---------- Load chatbot ----------
try:
    chatbot = load_chatbot()
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
                with st.expander("Technical details"):
                    st.code(traceback.format_exc(), language="python")
                st.stop()

        st.markdown(response)

        # ---- Metrics and context expanders ----
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
