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
os.environ["TQDM_DISABLE"] = "1"          # kill tqdm monitor thread
os.environ["TRANSFORMERS_VERBOSITY"] = "error"   # suppress HF warnings

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import faulthandler
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
import streamlit as st

from utils.config_loader import load_config
from utils.retriever import Retriever

hf_logging.set_verbosity_error()
faulthandler.enable(all_threads=True)

# ---------- Load components with checkpoints ----------
@st.cache_resource
def load_components():
    config = load_config()

    # ------------------- Use HF repo ID -------------------
    adapter_repo_id = "rahulboby/chatveritas-lora-adapter" 
    use_lora = True   # or read from config, but set True if you always use adapter

    # Tokenizer: load from adapter repo (which contains tokenizer files)
    # If tokenizer files aren't there, fallback to base model
    tokenizer = AutoTokenizer.from_pretrained(adapter_repo_id)

    # ------------------- Base model -------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],   # still from config, e.g. "Qwen/Qwen-7B"
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        max_memory={"cpu": "12GiB", 0: "4GiB"} if torch.cuda.is_available() else {"cpu": "12GiB"},
        low_cpu_mem_usage=True,
    )

    # ------------------- Load LoRA from HF -------------------
    if use_lora:
        model = PeftModel.from_pretrained(base_model, adapter_repo_id)
    else:
        model = base_model

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- 4. Load Retriever (after model) ----------
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer, retriever


# ---------- Generate with checkpoints ----------
def generate_response(question, model, tokenizer, retriever):
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

    # ---- Chat template ----
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ---- Tokenization ----
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # ---- Generation ----
    with torch.inference_mode():
        temperature = config["generation"]["temperature"]
        generation_kwargs = {
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        gen_start = time.perf_counter()
        outputs = model.generate(**inputs, **generation_kwargs)
        metrics["generation_time"] = time.perf_counter() - gen_start

    # ---- Decode ----
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    metrics["prompt_tokens"] = inputs["input_ids"].shape[1]

    return response.strip(), chunks, metrics


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Qwen_RAG Chatbot", layout="wide")
st.title("Qwen RAG Chatbot")


# We catch any exception during loading and display it in the UI
try:
    model, tokenizer, retriever = load_components()
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
                response, chunks, metrics = generate_response(prompt, model, tokenizer, retriever)
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