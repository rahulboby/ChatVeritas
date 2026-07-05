from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    ):

        self.model = CrossEncoder(
            model_name,
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
