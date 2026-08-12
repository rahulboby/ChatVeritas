"""
scripts/prepare_finetune_dataset.py

Creates a synthetic instruction-tuning dataset from the EXACT chunks
used in the vector database (chunks.pkl).

This ensures perfect alignment between fine‑tuning data and inference‑time retrieval.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.utils import CacheManager, QuestionGenerator
from configs import settings
from core.llm import resolve_llm_provider


def load_config() -> dict:
    """Compatibility loader for legacy JSON configs.

    Tests and some scripts patch this function; provide a simple
    implementation so callers can override via `patch.object`.
    """
    return {}


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    config = load_config()
    provider = resolve_llm_provider(settings_module=settings)

    # ---------- 2. Load the exact chunks from the vector DB ----------
    vectorstore_dir = PROJECT_ROOT / config.get("paths", {}).get("vectorstore", "vectorstore")
    chunks_path = vectorstore_dir / "chunks.pkl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.pkl not found at: {chunks_path}\n"
            "Please run the ingestion pipeline first to build the vector database."
        )

    print(f"Loading chunks from: {chunks_path}")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        raise ValueError("No chunks found in chunks.pkl. Check your ingestion pipeline.")

    # ---------- 3. Initialize generator and cache ----------
    generator = QuestionGenerator(
        api_key=provider.api_key,
        model=provider.model,
        base_url=provider.base_url,
        temperature=settings.QUESTION_GENERATION_TEMPERATURE,
        max_retries=settings.QUESTION_GENERATION_MAX_RETRIES,
        response_format=provider.response_format,
    )
    print(
        f"Using {provider.name} provider | model={provider.model} | "
        f"base_url={provider.base_url}"
    )

    cache_enabled = config.get("cache", {}).get("enabled", settings.CACHE_ENABLED)
    cache = None
    if cache_enabled:
        cache_dir = PROJECT_ROOT / config.get("cache", {}).get("directory", settings.CACHE_DIR)
        cache = CacheManager(cache_dir)

    # ---------- 4. Generate questions for each chunk ----------
    samples = []
    total_chunks = len(chunks)

    print(f"\nFound {total_chunks} chunks in vector database.\n")

    for chunk_meta in tqdm(chunks, desc="Generating Q&A pairs"):
        # Extract the actual text content
        chunk_text = chunk_meta["chunk"]  # adjust if your dict uses a different key, e.g., "text"
        source_file = chunk_meta.get("source", "unknown_source")
        chunk_id = chunk_meta.get("chunk_id", 0)

        if not chunk_text or len(chunk_text.strip()) < 20:
            continue  # skip very short / empty chunks

        # Generate questions (with cache)
        if cache is None:
            response = generator.generate(chunk_text)
        else:
            response = cache.get_or_create(
                chunk_text,
                generator.generate,
                validator=generator.validate_data,
            )

        topic = response["topic"]
        questions = response["questions"]

        # Create training samples (user -> assistant format)
        for question in questions:
            samples.append({
                "topic": topic,
                "source_file": source_file,
                "chunk_id": chunk_id,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": chunk_text},
                ],
            })

    if not samples:
        raise ValueError("No training samples were generated. Check your chunks and LLM provider.")

    # ---------- 5. Save dataset ----------
    processed_rel = config.get("paths", {}).get("processed_data", "processed/train.jsonl")
    output_file = PROJECT_ROOT / processed_rel
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    print("\n" + "-" * 40)
    print(f"Total Chunks Loaded : {total_chunks}")
    print(f"Training Samples     : {len(samples)}")
    if cache:
        print(f"Cache Entries        : {cache.count()}")
    print(f"Saved Dataset        : {output_file}")
    print("-" * 40)


if __name__ == "__main__":
    main()
