"""Purpose: Score and sort retrieved text chunks with a local cross-encoder model.
Dependencies: built-in: none; installed: sentence-transformers.
Custom: utils.model_store.
"""

from sentence_transformers import CrossEncoder
from utils.model_store import ensure_model_available


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    ):

        self.model = CrossEncoder(
            ensure_model_available(model_name),
            device=device
        )

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):

        pairs = [
            (
                query,
                item["chunk"]
            )
            for item in candidates
        ]

        scores = self.model.predict(
            pairs
        )

        for item, score in zip(
            candidates,
            scores
        ):
            item["rerank_score"] = float(score)

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return candidates[:top_k]
