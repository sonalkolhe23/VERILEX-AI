
from retrieval.embeddings import DenseRetriever


def test_dense_retrieval():
    chunks = [
        {
            "chunk_id": "chunk_1",
            "document_name": "GDPR.pdf",
            "page_number": 5,
            "article": "Article 5",
            "text": (
                "Personal data shall be processed lawfully, "
                "fairly and transparently."
            )
        },
        {
            "chunk_id": "chunk_2",
            "document_name": "GDPR.pdf",
            "page_number": 8,
            "article": "Article 6",
            "text": (
                "Processing is permitted when a valid legal "
                "basis exists, such as consent or contractual necessity."
            )
        },
        {
            "chunk_id": "chunk_3",
            "document_name": "GDPR.pdf",
            "page_number": 12,
            "article": "Article 9",
            "text": (
                "Special categories of personal information, "
                "including racial or ethnic origin, receive additional protection."
            )
        }
    ]

    retriever = DenseRetriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Under what circumstances can personal information be legally used?",
        top_k=3
    )

    assert len(results) == 3
    assert all("dense_score" in result for result in results)

    print("\nDense retrieval results:")

    for result in results:
        print(
            result["article"],
            round(result["dense_score"], 4),
            result["text"]
        )

    print("\nDense retrieval test passed!")


if __name__ == "__main__":
    test_dense_retrieval()
