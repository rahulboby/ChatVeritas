import pickle
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(
        self,
        index_path,
        chunks_path,
        embedding_model,
        top_k
    ):

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            chunks_path,
            "rb"
        ) as f:

            self.chunks = pickle.load(f)

        self.embedder = SentenceTransformer(
            embedding_model
        )

        self.top_k = top_k

    def retrieve(
        self,
        query
    ):

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            self.top_k
        )

        results = []

        for idx in indices[0]:
            results.append(
                self.chunks[idx]
            )

        return results