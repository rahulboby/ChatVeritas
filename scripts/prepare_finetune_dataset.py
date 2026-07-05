"""
scripts/prepare_finetune_dataset.py

Creates a synthetic instruction-tuning dataset from raw TXT files.

Pipeline

TXT Files
    ↓
Paragraph Chunking
    ↓
Sentence-aware Splitting
    ↓
Cache Lookup
    ↓
LLM Question Generation
    ↓
train.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from utils.cache import CacheManager
from utils.config_loader import load_config
from utils.paragraph_chunker import ParagraphChunker
from utils.question_generator import QuestionGenerator


def main():

    load_dotenv()

    config = load_config()

    provider = config["llm"].get("provider", "groq").lower()

    if provider != "groq":
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. Only 'groq' is supported."
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Create a .env file."
        )

    raw_dir = PROJECT_ROOT / config["paths"]["raw_data"]

    output_file = PROJECT_ROOT / config["paths"]["processed_data"]

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    chunker = ParagraphChunker(

        model_name=config["llm"]["tokenizer_model"],

        max_tokens=config["dataset"]["max_chunk_tokens"],

        min_paragraph_length=config["dataset"]["minimum_paragraph_length"]

    )

    generator = QuestionGenerator(

        api_key=api_key,

        model=config["llm"]["model"],

        temperature=config["llm"]["temperature"],

        max_retries=config["llm"]["max_retries"]

    )

    cache_enabled = config.get("cache", {}).get("enabled", True)
    cache = None

    if cache_enabled:
        cache = CacheManager(
            PROJECT_ROOT / config["cache"]["directory"]
        )

    samples = []

    txt_files = sorted(
        raw_dir.glob("*.txt")
    )

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {raw_dir}")

    print(f"\nFound {len(txt_files)} text files.\n")

    total_chunks = 0

    for txt_file in txt_files:

        print(f"Processing: {txt_file.name}")

        with open(
            txt_file,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        chunks = chunker.chunk_document(text)

        total_chunks += len(chunks)

        for chunk_id, chunk in enumerate(
            tqdm(
                chunks,
                desc=txt_file.name,
                leave=False
            ),
            start=1
        ):

            if cache is None:
                response = generator.generate(chunk)
            else:
                response = cache.get_or_create(
                    chunk,
                    generator.generate,
                    validator=generator.validate_data
                )

            topic = response["topic"]

            questions = response["questions"]

            for question in questions:

                sample = {

                    "topic": topic,

                    "source_file": txt_file.name,

                    "chunk_id": chunk_id,

                    "messages": [

                        {
                            "role": "user",
                            "content": question
                        },

                        {
                            "role": "assistant",
                            "content": chunk
                        }

                    ]

                }

                samples.append(sample)

    if total_chunks == 0:
        raise ValueError(
            "No eligible chunks were produced. Check the input files and "
            "dataset.minimum_paragraph_length."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for sample in samples:

            json.dump(
                sample,
                f,
                ensure_ascii=False
            )

            f.write("\n")

    print("\n------------------------------------")
    print(f"Processed Chunks : {total_chunks}")
    print(f"Training Samples : {len(samples)}")
    cache_entries = cache.count() if cache is not None else 0
    print(f"Cache Entries    : {cache_entries}")
    print(f"Saved Dataset    : {output_file}")
    print("------------------------------------")


if __name__ == "__main__":
    main()
