
# Used to test the retrieval of chunks from the FAISS index using a query.

import sys
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from utils.config_loader import load_config


def main():

    config = load_config()

    vectorstore_dir = (
        PROJECT_ROOT /
        config["paths"]["vectorstore"]
    )

    print("Loading embedding model...")

    embedder = SentenceTransformer(
        config["embedding"]["model"]
    )

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(
            vectorstore_dir /
            "index.faiss"
        )
    )

    with open(
        vectorstore_dir /
        "chunks.pkl",
        "rb"
    ) as f:

        chunks = pickle.load(f)

    while True:

        query = input(
            "\nQuestion: "
        )

        if query.lower() == "exit":
            break

        query_embedding = embedder.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            np.float32
        )

        distances, indices = index.search(
            query_embedding,
            config["retrieval"]["top_k"]
        )

        print("\nRetrieved Chunks:\n")

        for rank, idx in enumerate(indices[0]):

            print(
                f"--- Result {rank + 1} ---"
            )

            print(
                chunks[idx]
            )

            print()


if __name__ == "__main__":
    main()