import os
import sys
import time
import textwrap
import traceback
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

import faulthandler
import streamlit as st

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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

faulthandler.enable(all_threads=True)

# ---------- Load components with checkpoints ----------
@st.cache_resource
def load_components():

    config = load_config()
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


# ---------- Generate with checkpoints ----------
def generate_response(question, retriever):
    try:
        config = load_config()
    except Exception as e:
        raise

    # ---- Retrieval ----
    retrieval = retriever.retrieve(question)

    chunks = retrieval["results"]
    metrics = retrieval["metrics"]

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
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
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
        )
    except Exception as e:
        raise RuntimeError(f"Groq API request failed: {e}")

    metrics["generation_time"] = (
        time.perf_counter() - gen_start
    )

    response = completion.choices[0].message.content

    metrics["prompt_tokens"] = completion.usage.prompt_tokens

    return response.strip(), chunks, metrics


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Qwen_RAG Chatbot", layout="wide")
st.title("ChatVeritas: Fine-Tuned Two-Stage RAG Chatbot on Custom Dataset")

st.info(
    """
    **Deployment Notice**

    This public deployment uses the **Groq API** for language model inference.

    The original ChatVeritas research system includes a fine-tuned LoRA model.
    That model is not included here because its size exceeds the limits of free
    cloud deployment platforms.

    The complete retrieval pipeline—including FAISS retrieval, reranking,
    and context-grounded generation—remains unchanged.
    """
)


# We catch any exception during loading and display it in the UI
try:
    retriever = load_components()
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.code(traceback.format_exc(), language="python")
    st.stop()


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
        with st.spinner("Thinking..."):
            try:
                response, chunks, metrics = generate_response(prompt, retriever)
            except Exception as e:
                st.error(f"Error during generation: {e}")
                st.code(traceback.format_exc(), language="python")
                st.stop()

        st.markdown(response)

        # ... (metrics and context expanders – unchanged) ...
        with st.expander("RAG Metrics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding Time", f"{metrics['embedding_time_ms']:.2f} ms")
                st.metric("Retrieval Time", f"{metrics['retrieval_time_ms']:.2f} ms")
            with col2:
                st.metric("Re‑ranking Time", f"{metrics['reranking_time_ms']:.2f} ms")
                st.metric("Generation Time", f"{metrics['generation_time']:.2f} s")
                st.metric("Prompt Tokens", metrics["prompt_tokens"])
            with col3:
                st.metric("Retrieved Chunks", metrics["retrieved_chunks"])
                st.metric("Avg L2 Distance", f"{metrics['average_distance']:.3f}")

        with st.expander("Retrieved Context"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"### Chunk {i}")
                st.markdown(
                    f"**Source:** {chunk['source']}  \n"
                    f"**Chunk ID:** {chunk['chunk_id']}  \n"
                    f"**FAISS L2:** {chunk['distance']:.3f}  \n"
                    f"**Cross‑Encoder:** {chunk['rerank_score']:.3f}"
                )
                st.write(chunk["chunk"])
            st.markdown("### Sources Used")
            for source in metrics["sources"]:
                st.write(f"- {source}")

    st.session_state.messages.append({"role": "assistant", "content": response})