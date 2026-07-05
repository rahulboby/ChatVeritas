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

# ---------- Helper for debug logging ----------
def debug_log(msg):
    """Print a timestamped message to console (immediately flushed)."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)
    # Optionally also show in Streamlit UI – uncomment next line if needed
    # st.write(f"DEBUG: {msg}")

# ---------- Load components with checkpoints ----------
@st.cache_resource
def load_components():
    debug_log("START load_components")
    try:
        config = load_config()
        debug_log("Config loaded")
    except Exception as e:
        debug_log(f"Config load FAILED: {e}")
        raise

    adapter_path = Path(config["model"]["adapter_path"])
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path
    debug_log(f"Adapter path: {adapter_path}")

    use_lora = config["model"].get("use_lora", False)
    debug_log(f"use_lora = {use_lora}")

    # Check adapter files
    adapter_is_complete = (
        (adapter_path / "adapter_config.json").is_file()
        and (
            (adapter_path / "adapter_model.safetensors").is_file()
            or (adapter_path / "adapter_model.bin").is_file()
        )
    )
    if use_lora and not adapter_is_complete:
        raise FileNotFoundError(f"Adapter incomplete at {adapter_path}")

    tokenizer_source = (
        adapter_path
        if use_lora and (adapter_path / "tokenizer_config.json").is_file()
        else config["model"]["base_model"]
    )
    debug_log(f"Tokenizer source: {tokenizer_source}")

    # ---------- 1. Load tokenizer ----------
    debug_log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    debug_log("Tokenizer loaded")

    # ---------- 2. Load base model (NO offload) ----------
    inference_config = config.get("inference", {})
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    debug_log(f"Model dtype: {model_dtype}")
    max_memory = {"cpu": inference_config.get("max_cpu_memory", "12GiB")}
    if torch.cuda.is_available():
        max_memory[0] = inference_config.get("max_gpu_memory", "4GiB")
    debug_log(f"Max memory: {max_memory}")

    debug_log("Loading base model (without offload folder) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        dtype=model_dtype,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        # NO offload_folder / offload_state_dict – they cause Windows crashes
    )
    debug_log("Base model loaded")

    # ---------- 3. Apply LoRA if needed ----------
    if use_lora:
        debug_log("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        debug_log("LoRA adapter loaded")
    else:
        model = base_model
        debug_log("LoRA disabled")

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    debug_log("Model ready")

    # ---------- 4. Load Retriever (after model) ----------
    debug_log("Loading Retriever...")
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
    debug_log("Retriever loaded")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        debug_log("CUDA cache cleared")

    debug_log("COMPLETE load_components")
    return model, tokenizer, retriever


# ---------- Generate with checkpoints ----------
def generate_response(question, model, tokenizer, retriever):
    debug_log(f"START generate_response for: {question[:50]}...")
    try:
        config = load_config()
        debug_log("Config reloaded for generation")
    except Exception as e:
        debug_log(f"Config reload FAILED: {e}")
        raise

    # ---- Retrieval ----
    debug_log("Calling retriever.retrieve()...")
    retrieval = retriever.retrieve(question)
    debug_log("Retrieval completed")

    chunks = retrieval["results"]
    metrics = retrieval["metrics"]
    debug_log(f"Retrieved {len(chunks)} chunks")

    context = "\n\n".join(item["chunk"] for item in chunks)
    debug_log(f"Context length: {len(context)} chars")

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
    debug_log("Prompt built")

    # ---- Chat template ----
    debug_log("Applying chat template...")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    debug_log("Chat template applied")

    # ---- Tokenization ----
    debug_log("Tokenizing input...")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    debug_log(f"Tokenized: {inputs['input_ids'].shape[1]} tokens")

    # ---- Generation ----
    debug_log("Starting model.generate()...")
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
    debug_log("Generation completed")

    # ---- Decode ----
    debug_log("Decoding generated tokens...")
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    metrics["prompt_tokens"] = inputs["input_ids"].shape[1]
    debug_log("Decoding completed")

    debug_log("COMPLETE generate_response")
    return response.strip(), chunks, metrics


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Qwen_RAG Chatbot", layout="wide")
st.title("Qwen RAG Chatbot")

debug_log("Streamlit UI starting – loading components...")

# We catch any exception during loading and display it in the UI
try:
    model, tokenizer, retriever = load_components()
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.code(traceback.format_exc(), language="python")
    debug_log(f"load_components EXCEPTION: {e}")
    st.stop()

debug_log("Components loaded, UI ready")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    debug_log(f"User prompt: {prompt[:50]}...")

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
                debug_log(f"generate_response EXCEPTION: {e}")
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
    debug_log("Response added to session state")