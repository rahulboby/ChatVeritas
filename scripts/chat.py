import argparse
import sys
from pathlib import Path
import time
import textwrap

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ChatVeritas directly in the terminal without Streamlit."
    )
    parser.add_argument(
        "--prompt",
        help="Ask one question and exit. Omit this option for interactive chat."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override generation.max_new_tokens for this run."
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Do not print the retrieved chunk details."
    )
    return parser.parse_args()


def main():

    args = parse_args()

    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")

    overall_start = time.perf_counter()

    print("=" * 80)
    print("Loading configuration...")
    config = load_config()
    print("Done.")
    print("=" * 80)

    adapter_path = Path(config["model"]["adapter_path"])
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path

    use_lora = bool(config["model"].get("use_lora", False))
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
    print(f"Total startup time: {(time.perf_counter()-overall_start):.2f} sec")
    print("Inference mode: FP16 with automatic CPU/GPU offload")
    print(f"Embedding device: {config['embedding'].get('device', 'cpu')}")
    print(f"Reranker device: {config['reranker'].get('device', 'cpu')}")

    single_prompt = args.prompt is not None

    if single_prompt:
        print("\nRunning one-shot terminal test.\n")
    else:
        print("\nRAG Chat Ready")
        print("Type 'exit' to quit\n")

    while True:

        if single_prompt:
            question = args.prompt
            print(f"You: {question}")
        else:
            try:
                question = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

        question = question.strip()

        if not question:
            if single_prompt:
                raise ValueError("--prompt cannot be empty.")
            continue

        if question.lower() == "exit":
            break

        print("\nRetrieving...")

        start = time.perf_counter()

        retrieval = retriever.retrieve(question)

        print(f"Retriever finished in {(time.perf_counter()-start):.2f} sec")

        retrieved_chunks = retrieval["results"]
        metrics = retrieval["metrics"]

        print(metrics)

        context = "\n\n".join(
            [
                item["chunk"]
                for item in retrieved_chunks
            ]
        )

        print(f"Context length: {len(context)} chars")

        prompt = textwrap.dedent(f"""
            You are an expert technical assistant answering questions about the provided documents.

            Use the retrieved context as your PRIMARY source of information.

            Guidelines:

            1. Base your answer primarily on the provided context.

            2. If the answer is explicitly stated in the context, answer confidently.

            3. If the answer is not explicitly stated but can be reasonably inferred from the available information, clearly state that it is an inference.

            4. Only respond with:
            "I don't have enough information in the provided documents."
            if the context contains insufficient information.

            5. Never invent facts.

            Context:
            {context}

            Question:
            {question}

            Answer:
        """).strip()

        print("Applying chat template...")

        start = time.perf_counter()

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"Template applied in {(time.perf_counter()-start):.3f} sec")

        print("Tokenizing prompt...")

        start = time.perf_counter()

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(model.device)

        print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
        print(f"Tokenization took {(time.perf_counter()-start):.3f} sec")

        print("Generating...")

        start = time.perf_counter()

        with torch.inference_mode():

            temperature = config["generation"]["temperature"]

            generation_kwargs = {
                "max_new_tokens": (
                    args.max_new_tokens
                    if args.max_new_tokens is not None
                    else config["generation"]["max_new_tokens"]
                ),
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id
            }

            if temperature > 0:
                generation_kwargs["temperature"] = temperature

            outputs = model.generate(**inputs, **generation_kwargs)

        print(f"Generation took {(time.perf_counter()-start):.2f} sec")

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        print(f"Generated {generated_tokens.shape[0]} tokens")

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        print("\nAssistant:\n")
        print(response.strip())

        if not args.no_sources:
            print("\nRetrieved Chunks")
            print("-" * 80)

            for chunk in retrieved_chunks:

                print(f"Source      : {chunk['source']}")
                print(f"Chunk ID    : {chunk['chunk_id']}")
                print(f"FAISS Rank  : {chunk['faiss_rank']}")
                print(f"L2 Distance : {chunk['distance']:.4f}")
                print(f"CE Score    : {chunk['rerank_score']:.4f}")
                print("-" * 80)

        if single_prompt:
            break


if __name__ == "__main__":
    main()
