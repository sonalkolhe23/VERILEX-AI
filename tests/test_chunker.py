
from ingestion.pdf_loader import PDFLoader
from retrieval.chunker import LegalChunker


def test_gdpr_chunking():
    loader = PDFLoader()
    chunker = LegalChunker()

    document = loader.load(
        "data/uploads/sample_legal_document.pdf"
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 0

    # Every chunk must have its source information.
    for chunk in chunks:
        assert chunk["chunk_id"]
        assert chunk["document_id"]
        assert chunk["document_name"]
        assert chunk["page_number"] >= 1
        assert chunk["text"].strip()

    # Check that legal article metadata was detected.
    article_chunks = [
        chunk for chunk in chunks
        if chunk["article"] is not None
    ]

    assert len(article_chunks) > 0

    print("Total PDF pages:", document["total_pages"])
    print("Total chunks:", len(chunks))
    print("Chunks with article metadata:", len(article_chunks))
    print("Unique articles:", len({
        chunk["article"] for chunk in article_chunks
    }))
    print("First chunk:", chunks[0])



def test_article_spanning_multiple_pages_keeps_accurate_page_references():
    chunker = LegalChunker(
        max_chars=45,
        overlap_chars=0,
        min_chars=1
    )

    pages = [
        {
            "page_number": 1,
            "text": "Article 5\n" + "A" * 35
        },
        {
            "page_number": 2,
            "text": "B" * 35
        },
        {
            "page_number": 3,
            "text": "Article 6\n" + "C" * 20
        }
    ]

    chunks = chunker.chunk_pages(
        pages,
        document_id="test-document",
        document_name="Test Legal Document"
    )

    article_5_chunks = [
        chunk for chunk in chunks
        if chunk["article"] == "Article 5"
    ]

    article_6_chunks = [
        chunk for chunk in chunks
        if chunk["article"] == "Article 6"
    ]

    assert len(article_5_chunks) == 2
    assert article_5_chunks[0]["page_numbers"] == [1]
    assert article_5_chunks[1]["page_numbers"] == [2]

    assert len(article_6_chunks) == 1
    assert article_6_chunks[0]["page_numbers"] == [3]


def test_separate_articles_are_not_mixed():
    chunker = LegalChunker()

    pages = [
        {
            "page_number": 1,
            "text": (
                "Article 5\n"
                "Personal data shall be processed lawfully."
            )
        },
        {
            "page_number": 2,
            "text": (
                "Article 6\n"
                "Processing shall be lawful only if a condition applies."
            )
        }
    ]

    chunks = chunker.chunk_pages(
        pages,
        document_id="test-document",
        document_name="Test Legal Document"
    )

    article_5_chunks = [
        chunk for chunk in chunks
        if chunk["article"] == "Article 5"
    ]

    article_6_chunks = [
        chunk for chunk in chunks
        if chunk["article"] == "Article 6"
    ]

    assert len(article_5_chunks) == 1
    assert len(article_6_chunks) == 1

    assert "Article 6" not in article_5_chunks[0]["text"]
    assert "Article 5" not in article_6_chunks[0]["text"]



if __name__ == "__main__":
    test_gdpr_chunking()
    print("\nLegal chunking test passed!")
