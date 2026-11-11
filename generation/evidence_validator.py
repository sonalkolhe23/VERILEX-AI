
class EvidenceValidator:
    """Validates citations against the evidence supplied to the model."""

    @staticmethod
    def validate(citation_indices, evidence):
        """
        Keep only citation indices that refer to existing evidence.
        Indices are 1-based.
        """
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
