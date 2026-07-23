import os
import sys
import time
import textwrap
import traceback
from pathlib import Path

# ========== THREADING LIMITS (prevent tqdm & BLAS threads from crashing) ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import faulthandler
import torch
import streamlit as st
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

from utils.config_loader import load_config
from utils.retriever import Retriever

hf_logging.set_verbosity_error()
faulthandler.enable(all_threads=True)


# ---------- Cached configuration ----------
@st.cache_data
def get_config():
    """Load and cache the application configuration."""
    return load_config()


# ---------- Cached model loader (depends on `use_lora`) ----------
@st.cache_resource
def load_model_and_tokenizer(use_lora: bool):
    """
    Load the base model and optionally apply the LoRA adapter.
    Returns: (model, tokenizer)
    """
    config = get_config()
    base_model_name = config["model"]["base_model"]
    adapter_repo_id = config["model"]["adapter_repo_id"]

    # ---------- Load tokenizer ----------
    # If using LoRA, try to load tokenizer from the adapter repo (it might contain custom tokens).
    # Otherwise, fall back to the base model.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_repo_id if use_lora else base_model_name)
    except Exception:
        # If adapter repo doesn't have tokenizer files, use base model's tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # ---------- Load base model ----------
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    max_memory = {"cpu": "12GiB", 0: "4GiB"} if torch.cuda.is_available() else {"cpu": "12GiB"}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )

    # ---------- Apply LoRA if requested ----------
    if use_lora:
        try:
            model = PeftModel.from_pretrained(base_model, adapter_repo_id)
        except Exception as e:
            st.error(f"Failed to load LoRA adapter from `{adapter_repo_id}`: {e}")
            st.warning("Falling back to the base model without LoRA.")
            model = base_model
    else:
        model = base_model

    model.eval()

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Clear GPU cache after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer


# ---------- Cached retriever loader ----------
@st.cache_resource
def load_retriever():
    """Load the FAISS index and chunks."""
    config = get_config()
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


# ---------- Generation function ----------
def generate_response(question, model, tokenizer, retriever):
    """Retrieve context and generate an answer using the loaded model."""
    config = get_config()  # re-use cached config

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

    # ---- Apply chat template ----
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ---- Tokenization ----
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # ---- Generation settings ----
    temperature = config["generation"]["temperature"]
    generation_kwargs = {
        "max_new_tokens": config["generation"]["max_new_tokens"],
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    # ---- Generation ----
    with torch.inference_mode():
        gen_start = time.perf_counter()
        outputs = model.generate(**inputs, **generation_kwargs)
        metrics["generation_time"] = time.perf_counter() - gen_start

    # ---- Decode only the new tokens ----
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    metrics["prompt_tokens"] = inputs["input_ids"].shape[1]

    return response.strip(), chunks, metrics


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
        st.info("LoRA adapter will be loaded from the Hugging Face repo specified in `config.yaml`.")
    else:
        st.info("Using the base model only.")

    # Optional: display current model names from config
    config = get_config()
    st.caption(f"Base model: `{config['model']['base_model']}`")
    st.caption(f"Adapter repo: `{config['model']['adapter_repo_id']}`")

# ---------- Load components ----------
try:
    # Load retriever (independent of LoRA setting)
    retriever = load_retriever()
    # Load model + tokenizer – cache is keyed by `use_lora`, so it reloads when the checkbox changes.
    model, tokenizer = load_model_and_tokenizer(use_lora)
except Exception as e:
    st.error(f"Failed to load components: {e}")
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
                response, chunks, metrics = generate_response(prompt, model, tokenizer, retriever)
            except Exception as e:
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

    # Append assistant's message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})