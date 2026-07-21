import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from peft import PeftModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from utils.config_loader import load_config
from utils.retriever import Retriever


SYSTEM_PROMPT = """
You are ChatVeritas, an expert assistant for answering questions about {config['topic']}.

Instructions:

- Use the retrieved context as your primary source.
- Answer only using the provided context.
- If the answer is explicitly stated, answer confidently.
- If it can only be inferred, clearly state that it is an inference.
- If the answer is not contained in the retrieved context, reply:

'I don't have enough information in the provided documents.'

Never invent facts.
""".strip()

SMALL_TALK = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "thanks",
    "thank you",
    "bye"
}

def load_chatveritas(use_lora: bool = False):
    print("Loading configuration...")
    config = load_config()
    print("Done.")
    print("=" * 80)

    adapter_path = Path(config["model"]["adapter_path"])
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path

    adapter_is_complete = (
        (adapter_path / "adapter_config.json").is_file()
        and (
            (adapter_path / "adapter_model.safetensors").is_file()
            or (adapter_path / "adapter_model.bin").is_file()
        )
    )

    if use_lora and not adapter_is_complete:
        raise FileNotFoundError(
            f"A complete ChatVeritas adapter was not found at {adapter_path}. "
            "Run `python scripts/fine_tune.py` first, or disable model.use_lora."
        )

    tokenizer_source = (
        adapter_path
        if use_lora and (adapter_path / "tokenizer_config.json").is_file()
        else config["model"]["base_model"]
    )

    # --------------------------------------------------

    print("\nLoading tokenizer...")

    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source
    )

    print(f"Tokenizer loaded in {(time.perf_counter()-start):.2f} sec")

    # --------------------------------------------------

    print("\nLoading Qwen model...")

    start = time.perf_counter()

    inference_config = config.get("inference", {})
    model_dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    max_memory = {
        "cpu": inference_config.get("max_cpu_memory", "12GiB")
    }
    if torch.cuda.is_available():
        max_memory[0] = inference_config.get("max_gpu_memory", "4GiB")

    offload_directory = PROJECT_ROOT / inference_config.get(
        "offload_directory",
        "data/model_offload"
    )
    offload_directory.mkdir(parents=True, exist_ok=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        dtype=model_dtype,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_directory,
        offload_state_dict=True,
        low_cpu_mem_usage=True
    )

    # If use_lora is False (see the toggle above), this branch is skipped
    # and "model" is simply the base Qwen model with no adapter applied.
    if use_lora:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False
        )
    else:
        model = base_model
    

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Qwen loaded in {(time.perf_counter()-start):.2f} sec")

    # --------------------------------------------------

    print("\nCreating Retriever...")

    start = time.perf_counter()

    retriever = Retriever(
        index_path=(
            PROJECT_ROOT /
            config["paths"]["vectorstore"] /
            "index.faiss"
        ),
        chunks_path=(
            PROJECT_ROOT /
            config["paths"]["vectorstore"] /
            "chunks.pkl"
        ),
        embedding_model=config["embedding"]["model"],
        top_k=config["retrieval"]["top_k"],
        faiss_candidates=config["retrieval"]["faiss_candidates"],
        embedding_device=config["embedding"].get("device", "cpu"),
        reranker_model=config["reranker"]["model"],
        reranker_device=config["reranker"].get("device", "cpu")
    )

    print(f"Retriever created in {(time.perf_counter()-start):.2f} sec")

    # --------------------------------------------------

    print("\nEverything loaded.")
    print("Inference mode: FP16 with automatic CPU/GPU offload")
    print(f"Embedding device: {config['embedding'].get('device', 'cpu')}")
    print(f"Reranker device: {config['reranker'].get('device', 'cpu')}")

    return {
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "retriever": retriever
    }

def chat(chatveritas, question):

    config = chatveritas["config"]
    model = chatveritas["model"]
    tokenizer = chatveritas["tokenizer"]
    retriever = chatveritas["retriever"]
    system_prompt = SYSTEM_PROMPT.format(
        config=config
    )

    question = question.strip()

    if not question:
        return ""

    if question.lower() in SMALL_TALK:

        messages = [
            {
                "role": "user",
                "content": question
            }
        ]

    else:

        retrieval = retriever.retrieve(question)

        retrieved_chunks = retrieval["results"]

        context_parts = []

        for i, item in enumerate(retrieved_chunks, start=1):

            context_parts.append(
                f"""
Document {i}
Source: {item['source']}

{item['chunk']}
"""
            )

        context = "\n" + ("\n" + "=" * 80 + "\n").join(context_parts)

        user_content = f"""
Context:

{context}

Question:

{question}
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    temperature = config["generation"]["temperature"]

    generation_kwargs = {
        "max_new_tokens": config["generation"]["max_new_tokens"],
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id
    }

    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()

def main():

    overall_start = time.perf_counter()

    print("=" * 80)

    choice = input("Use LoRA? (y/n): ").lower().strip()

    use_lora = choice in ["y", "yes"]

    chatveritas = load_chatveritas(use_lora)

    print(f"Startup time: {time.perf_counter()-overall_start:.2f} sec")
    print("\nRAG Chat Ready")
    print("Type 'exit' to quit\n")

    while True:

        try:
            question = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\nExiting.")
            break

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = chat(chatveritas, question)

        print("\nAssistant:\n")
        print(answer)

if __name__ == "__main__":
    main()