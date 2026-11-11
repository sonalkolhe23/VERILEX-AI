from unittest.mock import patch

from generation.answer_generator import AnswerGenerator


def test_grounded_answer():
    generator = AnswerGenerator()

    mock_response = {
        "message": {
            "content": (
                '{"answer": "Personal data must be processed lawfully '
                'and fairly. [1]", "citation_indices": [1], '
                '"abstained": false}'
            )
        }
    }

    evidence = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 5,
            "text": "Personal data shall be processed lawfully and fairly.",
        }
    ]

    with patch.object(
        generator, "_generate_with_ollama", return_value=mock_response
    ):
        result = generator.generate(
            "What principles apply to personal data?", evidence
        )

    assert result["abstained"] is False
    assert "[1]" in result["answer"]
    assert result["citations"][0]["document_name"] == "GDPR.pdf"
    assert result["citations"][0]["page_number"] == 5


def test_structured_claims_generate_inline_citations():
    generator = AnswerGenerator()

    mock_response = {
        "message": {
            "content": (
                '{"claims": ['
                '{"text": "Personal data must be processed lawfully and fairly.", '
                '"citation_indices": [1], '
                '"supporting_quote": "Personal data shall be processed lawfully and fairly."},'
                '{"text": "Processing must ensure integrity and confidentiality.", '
                '"citation_indices": [2], '
                '"supporting_quote": "appropriate security"}'
                '], "abstained": false}'
            )
        }
    }

    evidence = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 6,
            "text": "Personal data shall be processed lawfully and fairly.",
        },
        {
            "document_name": "GDPR.pdf",
            "page_number": 7,
            "text": "Personal data shall be processed in a manner that ensures appropriate security.",
        },
    ]

    with patch.object(
        generator, "_generate_with_ollama", return_value=mock_response
    ):
        result = generator.generate("What principles apply?", evidence)

    assert result["abstained"] is False
    assert "[1]" in result["answer"]
    assert "[2]" in result["answer"]
    assert len(result["citations"]) == 2


def test_abstention_with_no_evidence():
    generator = AnswerGenerator()

    result = generator.generate(
        "What is the maximum prison sentence for tax fraud?", []
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_abstention_when_model_says_insufficient_evidence():
    generator = AnswerGenerator()

    mock_response = {
        "message": {
            "content": (
                '{"answer": "Insufficient evidence.", '
                '"citation_indices": [], "abstained": true}'
            )
        }
    }

    evidence = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 5,
            "text": "Personal data shall be processed lawfully.",
        }
    ]

    with patch.object(
        generator, "_generate_with_ollama", return_value=mock_response
    ):
        result = generator.generate(
            "What is the maximum prison sentence for tax fraud?", evidence
        )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_structured_claim_with_invalid_citation_is_rejected():
    generator = AnswerGenerator()

    mock_response = {
        "message": {
            "content": (
                '{"claims": ['
                '{"text": "This statement has no valid source.", '
                '"citation_indices": [99]}'
                '], "abstained": false}'
            )
        }
    }

    evidence = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 5,
            "text": "Personal data shall be processed lawfully.",
        }
    ]

    with patch.object(
        generator, "_generate_with_ollama", return_value=mock_response
    ):
        result = generator.generate("What does the document say?", evidence)

    assert result["abstained"] is True
    assert result["citations"] == []

def test_quote_validator_rejects_quote_not_in_evidence():
    from generation.evidence_validator import EvidenceValidator

    evidence_text = (
        "Personal data shall be processed lawfully and fairly."
    )

    assert EvidenceValidator.validate_quote(
        "processed lawfully and fairly", evidence_text
    ) is True

    assert EvidenceValidator.validate_quote(
        "Personal data must be stored forever", evidence_text
    ) is False

def test_claim_is_rejected_when_quote_does_not_match_cited_evidence():
    from generation.answer_generator import AnswerGenerator

    generator = AnswerGenerator()

    evidence = [
        {
            "document_name": "GDPR.pdf",
            "page_number": 6,
            "text": "Personal data shall be processed lawfully and fairly.",
        }
    ]

    claims = [
        {
            "text": "Personal data must be stored forever.",
            "citation_indices": [1],
            "supporting_quote": "Personal data must be stored forever.",
        }
    ]

    result = generator._process_claims(claims, evidence)

    assert result["abstained"] is True
    assert "cannot provide a reliable answer" in result["answer"]
    assert result["citations"] == []
