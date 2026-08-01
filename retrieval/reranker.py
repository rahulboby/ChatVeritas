"""
retrieval/reranker.py

Cross-encoder reranker for two-stage retrieval.

Wraps the sentence-transformers ``CrossEncoder`` and re-ranks a list of
candidate chunks by relevance to the query.  This is the project's own
custom implementation — no third-party orchestration framework is used.

Algorithm
---------
1. Form (query, chunk) pairs for every FAISS candidate.
2. Score all pairs in one batched forward pass.
3. Sort descending by score and return the top-k results.
"""

from sentence_transformers import CrossEncoder

from core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker.

    Parameters
    ----------
    model_name : str
        Sentence-transformers cross-encoder model identifier.
    device : str
        Device to run inference on (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        logger.info(
            "Loading cross-encoder reranker: %s (device=%s)", model_name, device
        )
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Reranker loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank candidate chunks using the cross-encoder.

        Parameters
        ----------
        query : str
            The user query.
        candidates : list[dict]
            Candidate chunks from FAISS, each containing at least a
            ``"chunk"`` key with the document text.
        top_k : int
            Number of top candidates to return.

        Returns
        -------
        list[dict]
            Top-k candidates sorted by cross-encoder score (descending).
            Each dict is augmented with a ``"rerank_score"`` key.
        """
        pairs = [(query, item["chunk"]) for item in candidates]

        scores = self.model.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            "Reranked %d candidates → returning top %d.", len(candidates), top_k
        )

        return candidates[:top_k]
