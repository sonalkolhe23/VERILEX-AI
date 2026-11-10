
import re

from rank_bm25 import BM25Okapi


class BM25Search:
    """Keyword-based retrieval over legal document chunks."""

    def __init__(self):
        self.chunks = []
        self.model = None

    @staticmethod
    def tokenize(text):
        """Normalize text into searchable tokens."""
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def fit(self, chunks):
        """Build the BM25 index from legal chunks."""
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection.")

        self.chunks = chunks

        tokenized_chunks = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.model = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=5):
        """Return the most relevant chunks for a query."""
        if self.model is None:
            raise RuntimeError("Run fit() before search().")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.model.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:
            if scores[index] <= 0:
                continue

            result = self.chunks[index].copy()
            result["bm25_score"] = float(scores[index])
            results.append(result)

        return results
