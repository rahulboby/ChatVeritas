import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from utils.config_loader import load_config
from utils.retriever import Retriever


def main():

    overall_start = time.perf_counter()

    print("=" * 80)
    print("Loading configuration...")
    config = load_config()
    print("Done.")
    print("=" * 80)

    # --------------------------------------------------

    print("\nLoading tokenizer...")

    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    print(f"Tokenizer loaded in {(time.perf_counter()-start):.2f} sec")

    # --------------------------------------------------

    print("\nLoading Qwen model...")

    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.float16,
        device_map="auto"
    )

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
        faiss_candidates=config["retrieval"]["faiss_candidates"]
    )

    print(f"Retriever created in {(time.perf_counter()-start):.2f} sec")

    # --------------------------------------------------

    print("\nEverything loaded.")
    print(f"Total startup time: {(time.perf_counter()-overall_start):.2f} sec")

    print("\nRAG Chat Ready")
    print("Type 'exit' to quit\n")

    while True:

        question = input("You: ")

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

        prompt = f"""
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
"""

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

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
                do_sample=True
            )

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

        print("\nRetrieved Chunks")
        print("-" * 80)

        for chunk in retrieved_chunks:

            print(f"Source      : {chunk['source']}")
            print(f"Chunk ID    : {chunk['chunk_id']}")
            print(f"FAISS Rank  : {chunk['faiss_rank']}")
            print(f"L2 Distance : {chunk['distance']:.4f}")
            print(f"CE Score    : {chunk['rerank_score']:.4f}")
            print("-" * 80)


if __name__ == "__main__":
    main()