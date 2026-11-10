
from retrieval.hybrid_search import HybridSearch


def test_hybrid_search():
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
                "Processing of special categories of personal "
                "data is prohibited except under specific conditions."
            )
        }
    ]

    search = HybridSearch()
    search.fit(chunks)

    results = search.search(
        "When is processing of personal data lawful?",
        top_k=3
    )

    assert len(results) > 0
    assert all("rrf_score" in result for result in results)
    
    # Verify that cross-encoder reranking produced scores.
    assert all(
        "rerank_score" in result
        for result in results
    )

    # Verify results are ordered by reranking score.
    assert results == sorted(
        results,
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    # Each result must come from at least one retrieval method.
    assert all(
        result.get("found_by_bm25")
        or result.get("found_by_dense")
        for result in results
    )

    print("\nHybrid retrieval results:")

    for result in results:
        print(
            result["article"],
            "RRF:", round(result["rrf_score"], 5),
            "Rerank:", round(result["rerank_score"], 5),
            "BM25:", result.get("bm25_score"),
            "Dense:", result.get("dense_score"),
            "Found by BM25:", result.get("found_by_bm25", False),
            "Found by Dense:", result.get("found_by_dense", False)
        )

    print("\nHybrid retrieval test passed!")


if __name__ == "__main__":
    test_hybrid_search()
