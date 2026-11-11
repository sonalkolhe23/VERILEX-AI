
class AbstentionHandler:
    """Handles cases where there is insufficient legal evidence."""

    @staticmethod
    def should_abstain(evidence, model_abstained=False, valid_citations=None):
        """
        Decide whether the system should abstain.

        Abstain if:
        - No evidence was retrieved.
        - The model indicates insufficient evidence.
        - No valid citations support the answer.
        """
        if not evidence:
            return True

        if model_abstained:
            return True

        if valid_citations is not None and not valid_citations:
            return True

        return False

    @staticmethod
    def get_abstention_response():
        return (
            "I cannot provide a reliable answer based on the "
            "available legal documents. The retrieved evidence "
            "is insufficient to support a conclusion."
        )
