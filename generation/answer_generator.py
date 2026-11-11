
import json
import urllib.request
import urllib.error

from generation.evidence_validator import EvidenceValidator
from generation.abstention import AbstentionHandler


class AnswerGenerator:
    """Generates evidence-grounded answers using a local Ollama model."""

    def __init__(
        self,
        model="qwen2.5:3b",
        base_url="http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.validator = EvidenceValidator()
        self.abstention = AbstentionHandler()

    def _generate_with_ollama(self, question, evidence):
        """Send the question and evidence to the local Ollama model."""

        evidence_text = ""

        for index, item in enumerate(evidence, start=1):
            evidence_text += (
                f"\n[Evidence {index}]\n"
                f"Document: {item['document_name']}\n"
                f"Page: {item['page_number']}\n"
                f"Text: {item['text']}\n"
            )

        prompt = f"""
You are VERILEX AI, an evidence-grounded legal document analyst.

Answer using ONLY the supplied evidence.

Rules:
1. Never use outside knowledge.
2. Never invent legal facts.
3. Cite evidence using numbered references such as [1].
4. If evidence is insufficient, abstain.
5. Distinguish document statements from conclusions.
6. Return valid JSON only.

Use this structure:
{{
    "answer": "Your answer with citations such as [1]",
    "citation_indices": [1],
    "abstained": false
}}

If evidence is insufficient:
{{
    "answer": "Insufficient evidence to answer this question.",
    "citation_indices": [],
    "abstained": true
}}

QUESTION:
{question}

EVIDENCE:
{evidence_text}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are VERILEX AI. Return only valid JSON "
                        "and follow the evidence-grounding rules."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0}
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))

        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running."
            ) from error

    def generate(self, question, evidence):
        """Generate an answer and validate its evidence citations."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not evidence:
            return self._abstain()

        try:
            result = self._generate_with_ollama(question, evidence)
            model_result = json.loads(result["message"]["content"])

        except (KeyError, json.JSONDecodeError, TypeError):
            return self._abstain()

        answer = model_result.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            return self._abstain()

        valid_citations = self.validator.validate(
            model_result.get("citation_indices"),
            evidence
        )

        should_abstain = self.abstention.should_abstain(
            evidence=evidence,
            model_abstained=model_result.get("abstained") is True,
            valid_citations=valid_citations
        )

        if should_abstain:
            return self._abstain()

        referenced = [
            index for index in valid_citations
            if f"[{index}]" in answer
        ]

        if not referenced:
            return self._abstain()

        citations = [
            {
                "citation_index": index,
                "document_name": evidence[index - 1]["document_name"],
                "page_number": evidence[index - 1]["page_number"]
            }
            for index in referenced
        ]

        return {
            "answer": answer,
            "citations": citations,
            "abstained": False
        }

    def _abstain(self):
        return {
            "answer": self.abstention.get_abstention_response(),
            "citations": [],
            "abstained": True
        }
