
from unittest.mock import Mock

import pytest

from analysis.document_comparator import DocumentComparator


@pytest.fixture
def comparator():
    manager = Mock()

    manager.get_documents.return_value = [
        {
            "document_id": "doc1",
            "document_name": "GDPR.pdf"
        },
        {
            "document_id": "doc2",
            "document_name": "AI_Act.pdf"
        }
    ]

    manager.search.return_value = [
        {
            "document_id": "doc1",
            "document_name": "GDPR.pdf",
            "page_number": 6,
            "article": "Article 5",
            "text": "Personal data shall be processed lawfully, "
                    "fairly and in a transparent manner. " * 5
        },
        {
            "document_id": "doc2",
            "document_name": "AI_Act.pdf",
            "page_number": 20,
            "article": "Article 10",
            "text": "Data governance and data protection "
                    "requirements apply to high-risk AI. " * 5
        }
    ]

    return DocumentComparator(manager)


def test_retrieves_evidence_from_both_documents(comparator):
    evidence = comparator._retrieve_evidence(
        "data protection"
    )

    assert len(evidence) == 2
    assert {item["document_id"] for item in evidence} == {
        "doc1", "doc2"
    }


def test_rejects_comparison_with_only_one_document(comparator):
    comparator.document_manager.get_documents.return_value = [
        {
            "document_id": "doc1",
            "document_name": "GDPR.pdf"
        }
    ]

    evidence = comparator._retrieve_evidence("data protection")

    assert evidence == []


def test_rejects_heading_only_evidence(comparator):
    comparator.document_manager.search.return_value = [
        {
            "document_id": "doc1",
            "document_name": "GDPR.pdf",
            "page_number": 6,
            "text": "CHAPTER I"
        },
        {
            "document_id": "doc2",
            "document_name": "AI_Act.pdf",
            "page_number": 20,
            "text": "SECTION 3"
        }
    ]

    evidence = comparator._retrieve_evidence("data protection")

    assert evidence == []


def test_validates_cross_document_citations(comparator):
    evidence = comparator._retrieve_evidence("data protection")

    result = comparator._validate_items(
        [
            {
                "claim": "Both documents address data protection.",
                "citation_indices": [1, 2]
            },
            {
                "claim": "Unsupported single-document comparison.",
                "citation_indices": [1]
            }
        ],
        evidence
    )

    assert len(result) == 1
    assert len(result[0]["citations"]) == 2


def test_rejects_invalid_citation_indices(comparator):
    evidence = comparator._retrieve_evidence("data protection")

    result = comparator._validate_items(
        [
            {
                "claim": "Invalid citation.",
                "citation_indices": [1, 99]
            }
        ],
        evidence
    )

    assert result == []


def test_comparison_abstains_without_cross_document_evidence(
    comparator
):
    comparator.document_manager.search.return_value = [
        {
            "document_id": "doc1",
            "document_name": "GDPR.pdf",
            "page_number": 6,
            "text": "Personal data shall be processed lawfully. " * 5
        }
    ]

    result = comparator.compare("data protection")

    assert result["abstained"] is True
    assert result["potential_conflicts"] == []


def test_empty_topic_is_rejected(comparator):
    with pytest.raises(ValueError, match="cannot be empty"):
        comparator.compare("   ")


def test_filters_statements_that_say_no_conflict_exists(comparator):
    item = {
        "claim": "There is no evidence of incompatibility.",
        "citations": []
    }

    assert comparator._is_non_conflict(item) is True


def test_keeps_genuine_potential_conflicts(comparator):
    item = {
        "claim": "The requirements may be incompatible.",
        "reason": "The two documents establish different obligations."
    }

    assert comparator._is_non_conflict(item) is False


def test_removes_duplicate_citations(comparator):
    evidence = comparator._retrieve_evidence("data protection")

    # Add a duplicate source from the first document.
    duplicate_evidence = evidence + [evidence[0].copy()]

    result = comparator._validate_items(
        [
            {
                "claim": "Both documents address data protection.",
                "citation_indices": [1, 2, 3]
            }
        ],
        duplicate_evidence
    )

    assert len(result) == 1
    assert len(result[0]["citations"]) == 2
