"""Cloud Chat page: Groq generation with local retrieval."""

import faulthandler
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import apply_thread_limits

apply_thread_limits()

import streamlit as st
from dotenv import load_dotenv

from configs import settings
from core.chatveritas import ChatVeritas
from core.logger import get_logger

try:
    faulthandler.enable(all_threads=True)
except (OSError, RuntimeError):
    pass
load_dotenv(PROJECT_ROOT / ".env")
logger = get_logger(__name__)

st.set_page_config(page_title="ChatVeritas Cloud Chat", page_icon="☁️", layout="wide")


@st.cache_resource(show_spinner="Loading ChatVeritas...")
def load_chatbot() -> ChatVeritas:
    return ChatVeritas(mode="deploy")


def display_result(response: str, chunks: list[dict], metrics: dict) -> None:
    st.markdown(response)
    with st.expander("RAG Metrics"):
        first, second, third = st.columns(3)
        with first:
            st.metric("Embedding Time", f"{metrics.get('embedding_time_ms', 0.0):.2f} ms")
            st.metric("Retrieval Time", f"{metrics.get('retrieval_time_ms', 0.0):.2f} ms")
        with second:
            st.metric("Re-ranking Time", f"{metrics.get('reranking_time_ms', 0.0):.2f} ms")
            st.metric("Generation Time", f"{metrics.get('generation_time', 0.0):.2f} s")
            st.metric("Prompt Tokens", metrics.get("prompt_tokens", 0))
        with third:
            st.metric("Retrieved Chunks", metrics.get("retrieved_chunks", len(chunks)))
            st.metric("Avg L2 Distance", f"{metrics.get('average_distance', 0.0):.3f}")
    with st.expander("Retrieved Context"):
        for number, chunk in enumerate(chunks, start=1):
            st.markdown(f"### Chunk {number}")
            st.markdown(
                f"**Source:** {chunk.get('source', 'Unknown')}  \n"
                f"**Chunk ID:** {chunk.get('chunk_id', 'N/A')}  \n"
                f"**FAISS L2:** {chunk.get('distance', 0.0):.3f}  \n"
                f"**Cross-Encoder:** {chunk.get('rerank_score', 0.0):.3f}"
            )
            st.write(chunk.get("chunk", ""))
        if not chunks:
            st.info("No relevant documents were retrieved.")


st.title("ChatVeritas: Cloud Chat")
st.info(
    "Cloud Chat uses Groq for generation and the same local FAISS + cross-encoder "
    "retrieval pipeline as Offline Chat. Set `GROQ_API_KEY` in `.env` before use."
)
with st.sidebar:
    st.header("Cloud Settings")
    st.caption("Provider: `Groq`")
    st.caption(f"Model: `{settings.GROQ_MODEL_ID}`")

    st.markdown("---")
    if st.button("Load Cloud Chat", help="Initialize the Groq backend only when you click this button."):
        st.session_state["cloud_load_requested"] = True
        st.session_state["cloud_loaded"] = False
        st.rerun()


if "cloud_load_requested" not in st.session_state:
    st.session_state["cloud_load_requested"] = False
if "cloud_loaded" not in st.session_state:
    st.session_state["cloud_loaded"] = False

chatbot = None
if st.session_state["cloud_load_requested"]:
    try:
        chatbot = load_chatbot()
        st.session_state["cloud_loaded"] = True
    except Exception as exc:
        st.session_state["cloud_load_requested"] = False
        st.session_state["cloud_loaded"] = False
        st.error(f"Failed to load ChatVeritas: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc(), language="python")
        st.stop()

if not st.session_state["cloud_loaded"]:
    st.info("Click Load Cloud Chat to initialize this page.")
    st.stop()

history_key = "cloud_messages"
if history_key not in st.session_state:
    st.session_state[history_key] = []
for message in st.session_state[history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your indexed documents..."):
    st.session_state[history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chatbot.ask(prompt)
            except Exception as exc:
                logger.error("Cloud generation failed: %s", exc, exc_info=True)
                st.error(f"Generation failed: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc(), language="python")
                st.stop()
        display_result(result["response"], result["chunks"], result["metrics"])
    st.session_state[history_key].append({"role": "assistant", "content": result["response"]})
