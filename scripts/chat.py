"""
scripts/chat.py - ChatVeritas: Terminal CLI for Two-Stage RAG Chatbot

Runs the identical two-stage RAG pipeline and API-based generation as app.py,
designed for terminal/CLI interaction with streaming responses and retrieval metrics.
Supports both cloud providers (Groq, OpenAI) and locally hosted OpenAI-compatible LLMs.
"""
import os
import sys
import time
import textwrap
from pathlib import Path

# ---- Set project root and adjust sys.path ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Ensure robust stdout encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from openai import OpenAI
from dotenv import load_dotenv

from utils.config_loader import load_config
from utils.retriever import Retriever

# ========== THREADING & ENVIRONMENT ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def init_chatveritas():
    load_dotenv()
    config = load_config()

    # Determine API key (supporting cloud providers like Groq or local OpenAI-compatible endpoints)
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    url = config.get("llm", {}).get("url", "")
    provider = config.get("llm", {}).get("provider", "no-provider-specified")

    if not api_key:
        if "localhost" in url or "127.0.0.1" in url or provider.lower() in ["local", "ollama", "vllm", "lmstudio"]:
            api_key = "local"
        else:
            raise RuntimeError("GROQ_API_KEY (or OPENAI_API_KEY) not found in environment variables.")

    client = OpenAI(
        api_key=api_key,
        base_url=url
    )

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

    return client, retriever, config


def generate_response(question, client, retriever, config):
    # ---- Retrieval ----
    retrieval = retriever.retrieve(question)
    chunks = retrieval["results"]
    metrics = retrieval["metrics"]

    # Build context (identical to app.py)
    context = "\n\n".join(item["chunk"] for item in chunks)

    # ---- Build prompt (identical to app.py) ----
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
        stream = client.chat.completions.create(
            model=config["llm"].get("model", "no-model-specified"),
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
            stream=True,
        )
    except Exception as e:
        print(f"\n[Error] API request failed: {e}")
        return

    print("\nAssistant: ", end="", flush=True)
    full_response = ""
    prompt_tokens = 0

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token
            print(token, end="", flush=True)

        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens

    generation_time = time.perf_counter() - gen_start
    print("\n")

    # ---- Display RAG Metrics & Sources (mirroring app.py) ----
    print("-" * 60)
    print("RAG Metrics:")
    print(f"  - Embedding Time : {metrics.get('embedding_time_ms', 0.0):.2f} ms")
    print(f"  - Retrieval Time : {metrics.get('retrieval_time_ms', 0.0):.2f} ms")
    print(f"  - Re-ranking Time: {metrics.get('reranking_time_ms', 0.0):.2f} ms")
    print(f"  - Generation Time: {generation_time:.2f} s")
    if prompt_tokens:
        print(f"  - Prompt Tokens  : {prompt_tokens}")
    print(f"  - Retrieved Chunks: {metrics.get('retrieved_chunks', len(chunks))}")
    print(f"  - Avg L2 Distance: {metrics.get('average_distance', 0.0):.3f}")

    sources = metrics.get("sources", [])
    if sources:
        print("\nSources Used:")
        for src in sources:
            print(f"  - {src}")
    print("-" * 60)


def main():
    print("=" * 60)
    print("ChatVeritas: Two-Stage RAG Chatbot (Terminal CLI)")
    print("=" * 60)
    print("Initializing components...")

    start = time.perf_counter()
    try:
        client, retriever, config = init_chatveritas()
    except Exception as e:
        print(f"[Error] Initialization failed: {e}")
        return

    print(f"Ready in {time.perf_counter() - start:.2f}s.")
    print(f"Provider: {config['llm'].get('provider')} | Model: {config['llm'].get('model')}")
    print("Type your question below (or 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        generate_response(question, client, retriever, config)
        print()


if __name__ == "__main__":
    main()