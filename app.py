import sys
from pathlib import Path

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
        top_k=config["retrieval"]["top_k"]
    )

    return model, tokenizer, retriever


def generate_response(
    question,
    model,
    tokenizer,
    retriever
):

    config = load_config()

    retrieved_chunks = retriever.retrieve(
        question
    )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
        You are a helpful assistant.

        You MUST answer only using the provided context.

        If the answer cannot be found in the context,
        reply exactly:

        "I don't have enough information in the provided documents."

        Do not make assumptions.
        Do not use outside knowledge.

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

    return response.strip(), retrieved_chunks


st.set_page_config(
    page_title="Qwen RAG Chatbot",
    page_icon="🤖",
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

            response, chunks = generate_response(
                prompt,
                model,
                tokenizer,
                retriever
            )

            st.markdown(response)

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

                    st.write(chunk)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )