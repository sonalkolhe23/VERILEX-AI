
from sentence_transformers import CrossEncoder

from retrieval.bm25_search import BM25Search
from retrieval.embeddings import DenseRetriever


class HybridSearch:
    """
    Combines BM25 and dense retrieval using RRF,
    then reranks candidates using a cross-encoder.
    """

    def __init__(
        self,
        rrf_k=60,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.rrf_k = rrf_k
        self.bm25 = BM25Search()
        self.dense = DenseRetriever(model_name)
        self.reranker = CrossEncoder(reranker_name)
        self.is_fitted = False

    def fit(self, chunks):
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection.")

        self.bm25.fit(chunks)
        self.dense.fit(chunks)
        self.is_fitted = True

    def search(self, query, top_k=5, candidate_k=20):
        if not self.is_fitted:
            raise RuntimeError("Run fit() before search().")

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive.")

        bm25_results = self.bm25.search(
            query, top_k=candidate_k
        )

        dense_results = self.dense.search(
            query, top_k=candidate_k
        )

        combined = {}

        def add_results(results, source):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                if chunk_id not in combined:
                    combined[chunk_id] = result.copy()
                    combined[chunk_id]["rrf_score"] = 0.0

                else:
                    for score_key in ("bm25_score", "dense_score"):
                        if score_key in result:
                            combined[chunk_id][score_key] = (
                                result[score_key]
                            )

                combined[chunk_id]["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                combined[chunk_id][f"found_by_{source}"] = True

        add_results(bm25_results, "bm25")
        add_results(dense_results, "dense")

        candidates = sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        # Evaluate query and passage together.
        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        rerank_scores = self.reranker.predict(pairs, batch_size=4)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked_results = sorted(
            candidates,
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]
