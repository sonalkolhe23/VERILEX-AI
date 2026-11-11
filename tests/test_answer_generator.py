
from unittest.mock import patch

from generation.answer_generator import AnswerGenerator


def test_grounded_answer():
    generator = AnswerGenerator()

    mock_response = {
        "message": {
            "content": (
                '{"answer": "Personal data must be processed '
                'lawfully and fairly. [1]", '
                '"citation_indices": [1], '
                '"abstained": false}'
            )
        }
    }

    evidence = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 5,
            "text": "Personal data shall be processed lawfully and fairly."
        }
    ]

    with patch.object(
        generator,
        "_generate_with_ollama",
        return_value=mock_response
    ):
        result = generator.generate(
            "What principles apply to personal data?",
            evidence
        )

    assert result["abstained"] is False
    assert result["citations"][0]["document_name"] == "GDPR.pdf"
    assert result["citations"][0]["page_number"] == 5


def test_abstention_with_no_evidence():
    generator = AnswerGenerator()

    result = generator.generate(
        "What is the maximum prison sentence for tax fraud?",
        []
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_abstention_when_model_says_insufficient_evidence():
    generator = AnswerGenerator()

    mock_response = {
        "message": {
            "content": (
                '{"answer": "Insufficient evidence.", '
                '"citation_indices": [], '
                '"abstained": true}'
            )
        }
    }

    evidence = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 5,
            "text": "Personal data shall be processed lawfully."
        }
    ]

    with patch.object(
        generator,
        "_generate_with_ollama",
        return_value=mock_response
    ):
        result = generator.generate(
            "What is the maximum prison sentence for tax fraud?",
            evidence
        )

    assert result["abstained"] is True
    assert result["citations"] == []
