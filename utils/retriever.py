import pickle

import time

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from utils.reranker import Reranker


class Retriever:

    def __init__(
        self,
        index_path,
        chunks_path,
        embedding_model,
        top_k,
        faiss_candidates
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
        self.faiss_candidates = faiss_candidates

        self.reranker = Reranker()

    def retrieve(
        self,
        query
    ):

        metrics = {}

        # ----------------------------
        # Embed Query
        # ----------------------------
        start = time.perf_counter()

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        )

        metrics["embedding_time_ms"] = (
            time.perf_counter() - start
        ) * 1000

        query_embedding = query_embedding.astype(
            np.float32
        )

        # ----------------------------
        # FAISS Search
        # ----------------------------
        start = time.perf_counter()

        distances, indices = self.index.search(
            query_embedding,
            self.faiss_candidates
        )

        metrics["retrieval_time_ms"] = (
            time.perf_counter() - start
        ) * 1000

        metrics["faiss_candidates"] = self.faiss_candidates

        results = []

        for rank, (distance, idx) in enumerate(
            zip(
                distances[0],
                indices[0]
            ),
            start=1
        ):

            item = self.chunks[idx]

            results.append(
                {
                    "chunk": item["chunk"],
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "distance": float(distance),
                    "faiss_rank": rank
                }
            )

        # ----------------------------
        # Cross-Encoder Re-ranking
        # ----------------------------
        start = time.perf_counter()

        results = self.reranker.rerank(
            query=query,
            candidates=results,
            top_k=self.top_k
        )

        metrics["reranking_time_ms"] = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------
        # Final Metrics
        # ----------------------------
        metrics["retrieved_chunks"] = len(
            results
        )

        metrics["average_distance"] = float(
            np.mean(
                [
                    item["distance"]
                    for item in results
                ]
            )
        )

        metrics["sources"] = sorted(
            {
                item["source"]
                for item in results
            }
        )

        return {
            "results": results,
            "metrics": metrics
        }