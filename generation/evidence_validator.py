import re


class EvidenceValidator:
    """Validates citation indices and supporting quotes against evidence."""

    @staticmethod
    def validate(citation_indices, evidence):
        if not isinstance(citation_indices, list):
            return []

        valid_indices = []

        for index in citation_indices:
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and 1 <= index <= len(evidence)
                and index not in valid_indices
            ):
                valid_indices.append(index)

        return valid_indices

    @staticmethod
    def validate_quote(quote, evidence_text):
        """Check whether a supporting quote appears in the evidence text."""
        if not isinstance(quote, str) or not quote.strip():
            return False

        if not isinstance(evidence_text, str):
            return False

        def normalize(value):
            value = value.casefold()
            value = re.sub(r"\u00ad\s*", "", value)
            return " ".join(value.split())

        return normalize(quote) in normalize(evidence_text)