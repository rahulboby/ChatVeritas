"""FAISS retrieval and cross-encoder reranking for ChatVeritas.

Native and model dependencies are imported inside constructors so importing the
flat module does not initialise the ML stack on a cloud-only code path.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from core.exceptions import RetrievalError
from core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Cross-encoder reranker for candidate document chunks."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker: %s (device=%s)", model_name, device)
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Score candidates and return the highest-ranked ``top_k`` entries."""
        if not candidates:
            return []
        pairs = [(query, item["chunk"]) for item in candidates]
        scores = self.model.predict(pairs)
        ranked = [
            {**item, "rerank_score": float(score)}
            for item, score in zip(candidates, scores)
        ]
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


class Retriever:
    """Two-stage dense retrieval: FAISS candidate search then reranking."""

    def __init__(
        self,
        index_path: str | Path,
        chunks_path: str | Path,
        embedding_model: str,
        top_k: int,
        faiss_candidates: int,
        embedding_device: str = "cpu",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_device: str = "cpu",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if faiss_candidates < top_k:
            raise ValueError("faiss_candidates must be greater than or equal to top_k.")

        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        index_path = Path(index_path)
        chunks_path = Path(chunks_path)
        if not index_path.is_file() or not chunks_path.is_file():
            raise RetrievalError(
                "Vector-store files are missing. Run 'python scripts/ingest.py' "
                "after adding documents to data/raw/."
            )
        try:
            self.index = faiss.read_index(str(index_path))
            with chunks_path.open("rb") as chunks_file:
                self.chunks = pickle.load(chunks_file)
        except Exception as exc:
            raise RetrievalError(f"Unable to load the vector store: {exc}") from exc
        if not isinstance(self.chunks, list):
            raise RetrievalError("chunks.pkl must contain a list of chunk dictionaries.")
        if self.index.ntotal != len(self.chunks):
            raise RetrievalError(
                "FAISS index and chunk metadata are out of sync: "
                f"{self.index.ntotal} vectors != {len(self.chunks)} chunks. "
                "Run 'python scripts/ingest.py' to rebuild both files together."
            )
        if self.index.ntotal == 0:
            raise RetrievalError("The FAISS index is empty. Run 'python scripts/ingest.py'.")

        logger.info("Loading embedding model: %s (device=%s)", embedding_model, embedding_device)
        self.embedder = SentenceTransformer(embedding_model, device=embedding_device)
        self.top_k = top_k
        self.faiss_candidates = faiss_candidates
        self.reranker = Reranker(model_name=reranker_model, device=reranker_device)

    def retrieve(self, query: str) -> dict[str, Any]:
        """Retrieve and rerank chunks relevant to a non-empty query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        metrics: dict[str, Any] = {}

        started = time.perf_counter()
        embedding = self.embedder.encode([query], convert_to_numpy=True)
        metrics["embedding_time_ms"] = (time.perf_counter() - started) * 1000
        embedding = embedding.astype(self._np.float32)

        candidate_count = min(self.faiss_candidates, self.index.ntotal)
        started = time.perf_counter()
        distances, indices = self.index.search(embedding, candidate_count)
        metrics["retrieval_time_ms"] = (time.perf_counter() - started) * 1000
        metrics["faiss_candidates"] = candidate_count

        candidates = []
        for rank, (distance, index) in enumerate(zip(distances[0], indices[0]), start=1):
            if index < 0 or index >= len(self.chunks):
                continue
            item = self.chunks[index]
            if not isinstance(item, dict) or "chunk" not in item:
                raise RetrievalError("Each stored chunk must contain a 'chunk' field.")
            candidates.append(
                {
                    "chunk": item["chunk"],
                    "source": item.get("source", "unknown"),
                    "chunk_id": item.get("chunk_id", index),
                    "distance": float(distance),
                    "faiss_rank": rank,
                }
            )

        started = time.perf_counter()
        results = self.reranker.rerank(query, candidates, self.top_k)
        metrics["reranking_time_ms"] = (time.perf_counter() - started) * 1000
        metrics["retrieved_chunks"] = len(results)
        metrics["average_distance"] = (
            float(self._np.mean([item["distance"] for item in results])) if results else 0.0
        )
        metrics["sources"] = sorted({str(item["source"]) for item in results})
        return {"results": results, "metrics": metrics}
