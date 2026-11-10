
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


if __name__ == "__main__":
    test_gdpr_chunking()
    print("\nLegal chunking test passed!")
