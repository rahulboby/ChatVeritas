import sys
from pathlib import Path

import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from utils.config_loader import load_config
from utils.retriever import Retriever


@st.cache_resource
def load_components():

    config = load_config()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

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

    return model, tokenizer, retriever


def generate_response(
    question,
    model,
    tokenizer,
    retriever
):

    config = load_config()

    retrieval = retriever.retrieve(question)

    retrieved_chunks = retrieval["results"]

    metrics = retrieval["metrics"]

    context = "\n\n".join(
        [
            item["chunk"]
            for item in retrieved_chunks
        ]
    )

    prompt = f"""
        You are an expert technical assistant answering questions about the provided documents.

        Use the retrieved context as your PRIMARY source of information.

        Guidelines:

        1. Base your answer primarily on the provided context.

        2. If the answer is explicitly stated in the context, answer confidently.

        3. If the answer is not explicitly stated but can be reasonably inferred from the available information, clearly state that it is an inference.
        Examples:
        - "Based on the provided documents..."
        - "It appears that..."
        - "Although not explicitly mentioned..."

        4. Only respond with:
        "I don't have enough information in the provided documents."
        if the context contains insufficient information to reasonably answer or infer the answer.

        5. Never invent technologies, modules, libraries, or facts that are not supported by the context.

        6. Do not use outside knowledge unless it is necessary for basic reasoning.

        7. Prefer complete, technical answers over one-word responses.

        8. When appropriate, briefly explain why you reached your conclusion using evidence from the context.

        9. For technical questions:
        - Answer directly in the first sentence.
        - Then explain your reasoning.
        - If your answer is an inference, explicitly mention that.

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
        generation_start = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=config["generation"]["max_new_tokens"],
            temperature=config["generation"]["temperature"],
            do_sample=True
        )
    
        metrics["generation_time"] = (time.perf_counter() - generation_start)

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    metrics["prompt_tokens"] = inputs["input_ids"].shape[1]

    return (
        response.strip(),
        retrieved_chunks,
        metrics
    )


st.set_page_config(
    page_title="Qwen_RAG CHATBOT",
    layout="wide"
)

st.title("Qwen RAG Chatbot")

model, tokenizer, retriever = load_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

prompt = st.chat_input(
    "Ask a question..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Retrieving context..."):

            response, chunks, metrics = generate_response(
                prompt,
                model,
                tokenizer,
                retriever
            )

            st.markdown(response)

            with st.expander("RAG Metrics"):

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Embedding Time",
                        f"{metrics['embedding_time_ms']:.2f} ms"
                    )

                    st.metric(
                        "Retrieval Time",
                        f"{metrics['retrieval_time_ms']:.2f} ms"
                    )

                with col2:

                    st.metric(
                        "Re-ranking Time",
                        f"{metrics['reranking_time_ms']:.2f} ms"
                    )

                    st.metric(
                        "Generation Time",
                        f"{metrics['generation_time']:.2f} sec"
                    )

                    st.metric(
                        "Prompt Tokens",
                        metrics["prompt_tokens"]
                    )

                with col3:

                    st.metric(
                        "Retrieved Chunks",
                        metrics["retrieved_chunks"]
                    )

                    st.metric(
                        "Average L2 Distance",
                        f"{metrics['average_distance']:.3f}"
                    )

                with st.expander(
                    "Retrieved Context"
                ):

                    for i, chunk in enumerate(
                        chunks,
                        start=1
                    ):

                        st.markdown(
                            f"### Chunk {i}"
                        )

                        st.markdown(
                            f"""
                            **Source:** {chunk['source']}

                            **Chunk ID:** {chunk['chunk_id']}

                            **FAISS L2 Distance:** {chunk['distance']:.3f}

                            **Cross-Encoder Score:** {chunk['rerank_score']:.3f}
                            """
                        )

                        st.write(
                            chunk["chunk"]
                        )
                    st.markdown("### Sources Used")

                    for source in metrics["sources"]:
                        st.write(f"- {source}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )