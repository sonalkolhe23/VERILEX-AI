
from unittest.mock import Mock

import pytest

from ingestion.document_manager import DocumentManager


@pytest.fixture
def manager():
    pdf_loader = Mock()
    search_engine = Mock()

    pdf_loader.load.side_effect = lambda path: {
        "document_name": path.name,
        "document_path": str(path),
        "total_pages": 1,
        "metadata": {},
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "Article 5\n"
                    "Personal data shall be processed "
                    "lawfully and fairly."
                )
            }
        ]
    }

    return DocumentManager(
        pdf_loader=pdf_loader,
        search_engine=search_engine
    )


def test_add_multiple_documents(manager, tmp_path):
    pdf1 = tmp_path / "GDPR.pdf"
    pdf2 = tmp_path / "Contract.pdf"

    pdf1.write_bytes(b"first legal document")
    pdf2.write_bytes(b"second legal document")

    results = manager.add_documents([pdf1, pdf2])

    assert len(results) == 2
    assert len(manager.documents) == 2
    assert len(manager.chunks) == 2

    assert (
        manager.chunks[0]["document_id"]
        != manager.chunks[1]["document_id"]
    )


def test_document_identity_and_page_references(
    manager,
    tmp_path
):
    pdf = tmp_path / "GDPR.pdf"
    pdf.write_bytes(b"legal document content")

    result = manager.add_document(pdf)

    chunk = manager.chunks[0]

    assert chunk["document_id"] == result["document_id"]
    assert chunk["document_name"] == "GDPR.pdf"
    assert chunk["page_number"] == 1
    assert chunk["article"] == "Article 5"


def test_duplicate_document_is_rejected(manager, tmp_path):
    pdf = tmp_path / "GDPR.pdf"
    pdf.write_bytes(b"same legal document")

    manager.add_document(pdf)

    with pytest.raises(ValueError, match="already uploaded"):
        manager.add_document(pdf)


def test_missing_pdf_is_rejected(manager, tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        manager.add_document(missing_pdf)


def test_index_and_search_across_documents(
    manager,
    tmp_path
):
    pdf1 = tmp_path / "GDPR.pdf"
    pdf2 = tmp_path / "Contract.pdf"

    pdf1.write_bytes(b"first document")
    pdf2.write_bytes(b"second document")

    manager.add_documents([pdf1, pdf2])

    total_chunks = manager.build_index()

    assert total_chunks == 2
    assert manager.is_indexed is True

    manager.search_engine.search.return_value = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 1,
            "text": "Personal data shall be processed lawfully."
        }
    ]

    results = manager.search(
        "How should personal data be processed?"
    )

    assert len(results) == 1
    assert results[0]["document_name"] == "GDPR.pdf"

    manager.search_engine.fit.assert_called_once_with(
        manager.chunks
    )


def test_search_requires_index(manager):
    with pytest.raises(RuntimeError, match="Build the document index"):
        manager.search("What does the document say?")
