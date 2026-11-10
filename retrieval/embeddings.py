
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Semantic retrieval using transformer embeddings."""

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def fit(self, chunks):
        """Generate and store embeddings for legal chunks."""
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection.")

        self.chunks = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def search(self, query, top_k=5):
        """Retrieve chunks using cosine similarity."""
        if self.embeddings is None:
            raise RuntimeError("Run fit() before search().")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            result = self.chunks[int(index)].copy()
            result["dense_score"] = float(scores[index])
            results.append(result)

        return results
