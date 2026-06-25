import sys
from pathlib import Path

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

    config = load_config()

    print("Loading Qwen...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.float16,
        device_map="auto"
    )

    print("Loading Retriever...")

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
        top_k=config["retrieval"]["top_k"]
    )

    print("\nRAG Chat Ready")
    print("Type 'exit' to quit\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        retrieved_chunks = retriever.retrieve(
            question
        )

        context = "\n\n".join(
            retrieved_chunks
        )

        # Generate response using the model - by prompting it with the retrieved context and the question
        prompt = f"""
            You are a helpful AI assistant.

            Answer ONLY using the provided context.

            If the answer is not contained in the context,
            respond with:

            "I don't have enough information in the provided documents."

            Keep the answer concise and factual.

            Context:
            {context}

            Question:
            {question}

            Answer:
        """

        messages = [
            {
                "role": "user",
                "content": prompt
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

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
                do_sample=True
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        print("\nAssistant:")
        print(response.strip())
        print()


if __name__ == "__main__":
    main()