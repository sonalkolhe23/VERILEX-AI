
from retrieval.bm25_search import BM25Search


def test_bm25_search():
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
                "Processing shall be lawful only if "
                "at least one legal basis applies."
            )
        },
        {
            "chunk_id": "chunk_3",
            "document_name": "GDPR.pdf",
            "page_number": 12,
            "article": "Article 9",
            "text": (
                "Processing of personal data revealing "
                "racial or ethnic origin is prohibited."
            )
        }
    ]

    search = BM25Search()
    search.fit(chunks)

    results = search.search(
        "When is processing of personal data lawful?",
        top_k=2
    )

    assert len(results) > 0
    assert any(
    result["chunk_id"] == "chunk_2"
    for result in results
   )
    assert "bm25_score" in results[0]

    print("BM25 search results:")

    for result in results:
        print(
            result["article"],
            result["bm25_score"],
            result["text"]
        )

    print("\nBM25 test passed!")


if __name__ == "__main__":
    test_bm25_search()
