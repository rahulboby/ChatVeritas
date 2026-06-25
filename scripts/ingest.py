import sys
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from utils.config_loader import load_config


def chunk_text(
    text,
    chunk_size=500,
    overlap=50
):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += chunk_size - overlap

    return chunks


def main():

    config = load_config()

    raw_dir = (
        PROJECT_ROOT /
        config["paths"]["raw_data"]
    )

    vectorstore_dir = (
        PROJECT_ROOT /
        config["paths"]["vectorstore"]
    )

    vectorstore_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading embedding model...")

    embedder = SentenceTransformer(
        config["embedding"]["model"]
    )

    all_chunks = []

    txt_files = list(
        raw_dir.glob("*.txt")
    )

    print(
        f"Found {len(txt_files)} files"
    )

    for file in txt_files:

        print(f"Reading {file.name}")

        text = file.read_text(
            encoding="utf-8"
        )

        # Also save the source files with the chunks
        source_file = file.name
        
        chunks = chunk_text(
            text=text,
            chunk_size=config["retrieval"]["chunk_size"],
            overlap=config["retrieval"]["chunk_overlap"]
        )

        all_chunks.extend(chunks)

    print(
        f"Created {len(all_chunks)} chunks"
    )

    print("Generating embeddings...")

    embeddings = embedder.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        str(
            vectorstore_dir /
            "index.faiss"
        )
    )

    with open(
        vectorstore_dir /
        "chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            all_chunks,
            f
        )

    print(
        f"Saved {len(all_chunks)} chunks"
    )


if __name__ == "__main__":
    main()