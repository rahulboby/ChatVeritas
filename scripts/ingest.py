import sys
from pathlib import Path
import pickle
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from utils.config_loader import load_config
config = load_config()

def clean_text(text):
    # Remove repeated separator lines
    text = re.sub(r"^[=\-_*]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    return text

def chunk_text(
    text,
    chunk_size=500,
    overlap=50
):
    from utils.paragraph_chunker import ParagraphChunker

    chunker = ParagraphChunker(
        model_name=config["model"]["base_model"],
        max_tokens=config["dataset"]["max_chunk_tokens"]
    )

    chunks = chunker.chunk_document(text)
    return chunks

def main():

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
    chunk_id = 0

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

        # Preprocessing
        text = clean_text(text)

        # Also save the source files with the chunks
        
        chunks = chunk_text(
            text=text,
            chunk_size=config["retrieval"]["chunk_size"],
            overlap=config["retrieval"]["chunk_overlap"]
        )

        for chunk in chunks:
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": file.name,
                    "chunk": chunk
                }
            )

            chunk_id += 1

    print(
        f"Created {len(all_chunks)} chunks"
    )

    print("Generating embeddings...")

    texts = [item["chunk"] for item in all_chunks]
    embeddings = embedder.encode(
        texts,
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