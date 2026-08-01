"""
retrieval/retriever.py

Two-stage dense retriever: FAISS approximate nearest-neighbour search
followed by cross-encoder reranking.

Pipeline
--------
Query
    ↓ SentenceTransformer.encode()
Query embedding (float32)
    ↓ faiss.Index.search()
Top-N FAISS candidates (by L2 distance)
    ↓ Reranker.rerank()
Top-K results (by cross-encoder score)
"""

import pickle
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core.exceptions import RetrievalError
from core.logger import get_logger
from retrieval.reranker import Reranker

logger = get_logger(__name__)


class Retriever:
    """
    Two-stage retriever backed by a FAISS flat L2 index.

    Parameters
    ----------
    index_path : path-like
        Path to the serialised FAISS index (``index.faiss``).
    chunks_path : path-like
        Path to the pickled chunk metadata list (``chunks.pkl``).
    embedding_model : str
        Sentence-transformers model identifier used for query embedding.
    top_k : int
        Final number of chunks to return after reranking.
    faiss_candidates : int
        Number of candidates to retrieve from FAISS before reranking.
        Must be >= ``top_k``.
    embedding_device : str
        Device for the embedding model (``"cpu"`` or ``"cuda"``).
    reranker_model : str
        Cross-encoder model identifier.
    reranker_device : str
        Device for the reranker (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        index_path,
        chunks_path,
        embedding_model: str,
        top_k: int,
        faiss_candidates: int,
        embedding_device: str = "cpu",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_device: str = "cpu",
    ) -> None:
        logger.info("Loading FAISS index from: %s", index_path)
        self.index = faiss.read_index(str(index_path))

        logger.info("Loading chunk metadata from: %s", chunks_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(
            "Loading embedding model: %s (device=%s)", embedding_model, embedding_device
        )
        self.embedder = SentenceTransformer(embedding_model, device=embedding_device)

        self.top_k = top_k
        self.faiss_candidates = faiss_candidates

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.faiss_candidates <= 0:
            raise ValueError("faiss_candidates must be greater than zero.")

        if self.index.ntotal != len(self.chunks):
            raise RetrievalError(
                "FAISS index and chunk metadata are out of sync: "
                f"{self.index.ntotal} vectors != {len(self.chunks)} chunks."
            )

        if self.index.ntotal == 0:
            raise RetrievalError(
                "The FAISS index is empty. Run scripts/ingest.py first."
            )

        logger.info(
            "Retriever ready — index: %d chunks | top_k=%d | faiss_candidates=%d",
            self.index.ntotal,
            self.top_k,
            self.faiss_candidates,
        )

        self.reranker = Reranker(
            model_name=reranker_model, device=reranker_device
        )

    def retrieve(self, query: str) -> dict:
        """
        Retrieve the most relevant chunks for a query via two-stage retrieval.

        Stage 1 — Dense FAISS search:
            Embeds the query and retrieves ``faiss_candidates`` nearest
            neighbours by L2 distance.

        Stage 2 — Cross-encoder reranking:
            Scores all candidates with the cross-encoder and returns the
            top ``top_k`` results.

        Parameters
        ----------
        query : str
            The user question.

        Returns
        -------
        dict
            ``{"results": list[dict], "metrics": dict}``

            Each result dict contains:
                - ``chunk``        — document text
                - ``source``       — source filename
                - ``chunk_id``     — integer chunk index
                - ``distance``     — FAISS L2 distance
                - ``faiss_rank``   — rank in FAISS results (1-based)
                - ``rerank_score`` — cross-encoder score

            Metrics dict contains:
                - ``embedding_time_ms``  — query embedding time (ms)
                - ``retrieval_time_ms``  — FAISS search time (ms)
                - ``reranking_time_ms``  — reranking time (ms)
                - ``faiss_candidates``   — actual number of FAISS candidates
                - ``retrieved_chunks``   — number of returned chunks
                - ``average_distance``   — mean L2 distance of returned chunks
                - ``sources``            — sorted list of unique source files
        """
        logger.debug("Retrieving context for query: %r", query)
        metrics: dict = {}

        # ----------------------------
        # Stage 1a: Embed Query
        # ----------------------------
        start = time.perf_counter()

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        metrics["embedding_time_ms"] = (time.perf_counter() - start) * 1000

        query_embedding = query_embedding.astype(np.float32)

        # ----------------------------
        # Stage 1b: FAISS Search
        # ----------------------------
        start = time.perf_counter()

        candidate_count = min(self.faiss_candidates, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, candidate_count)

        metrics["retrieval_time_ms"] = (time.perf_counter() - start) * 1000
        metrics["faiss_candidates"] = candidate_count

        results = []

        for rank, (distance, idx) in enumerate(
            zip(distances[0], indices[0]), start=1
        ):
            item = self.chunks[idx]
            results.append(
                {
                    "chunk": item["chunk"],
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "distance": float(distance),
                    "faiss_rank": rank,
                }
            )

        # ----------------------------
        # Stage 2: Cross-Encoder Reranking
        # ----------------------------
        start = time.perf_counter()

        results = self.reranker.rerank(
            query=query, candidates=results, top_k=self.top_k
        )

        metrics["reranking_time_ms"] = (time.perf_counter() - start) * 1000

        # ----------------------------
        # Final Metrics
        # ----------------------------
        metrics["retrieved_chunks"] = len(results)
        metrics["average_distance"] = float(
            np.mean([item["distance"] for item in results])
        )
        metrics["sources"] = sorted({item["source"] for item in results})

        logger.debug(
            "Retrieval done — %d chunks | embed %.1f ms | faiss %.1f ms | rerank %.1f ms",
            metrics["retrieved_chunks"],
            metrics["embedding_time_ms"],
            metrics["retrieval_time_ms"],
            metrics["reranking_time_ms"],
        )

        return {"results": results, "metrics": metrics}
