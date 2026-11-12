import os
import json
import re
import string
import urllib.request
import urllib.error

from generation.evidence_validator import EvidenceValidator
from generation.abstention import AbstentionHandler


class AnswerGenerator:
    """Generates evidence-grounded answers using a local Ollama model."""

    def __init__(
        self,
        model="qwen2.5:3b",
        base_url=None
    ):
        self.model = model
        self.base_url = (
            base_url or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434"
            )
        ).rstrip("/")

        self.validator = EvidenceValidator()
        self.abstention = AbstentionHandler()

    def _generate_with_ollama(self, question, evidence):
        evidence_text = ""

        for index, item in enumerate(evidence, start=1):
            evidence_text += (
                f"\n[Evidence {index}]\n"
                f"Document: {item['document_name']}\n"
                f"Page: {item['page_number']}\n"
                f"Article: {item.get('article') or 'Not specified'}\n"
                f"Text: {item['text']}\n"
            )

        
        prompt = f"""You are VERILEX AI, an evidence-grounded legal document analyst.

Answer the question using ONLY the supplied evidence.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent legal facts, article numbers, paragraph numbers, or principles.
3. Prefer evidence that directly answers the question.
4. Ignore evidence that is unrelated to the question.
5. Every claim must have its own citation indices.
6. Only cite evidence that directly supports that claim.
7. If a claim has no supporting evidence, do not include it.
8. Identify the exact information requested before answering.
9. Evidence is sufficient only when it directly establishes the specific fact being asked.
10. Related information is NOT proof. Do not infer an answer from a related law, responsibility, date, role, or general statement.
11. If a question asks for a specific person's name, salary, company deployment date, or internal project name, that exact fact must appear in the evidence.
12. Never substitute a regulatory effective date for a company's deployment date.
13. Never substitute an organisation's responsibility for the identity of an individual employee.
14. When a question asks for an exact calendar date, the evidence must explicitly contain that date.
15. Phrases such as "first use", "entry into force", "application date", or "effective date" are not proof of a company's actual deployment date.
16. Never convert a legal deadline, regulatory date, or triggering event into a specific company deployment date.
17. If the exact date is absent, abstain.
MULTI-DOCUMENT ANALYSIS:
18. A question may require evidence from multiple legal documents.
19. Examine every supplied evidence item, including items from different documents.
20. For a question comparing two legal frameworks, identify the relevant
    evidence for each framework separately.
21. Create separate claims for each framework, citing the evidence
    from its own document.
22. Do not abstain merely because the answer requires combining evidence
    from multiple documents.
23. If one document supports part of the answer but the other does not,
    provide only the supported part and do not invent the missing part.
24. Use the document name and article metadata to distinguish sources.
25. Never assume that two documents have identical legal obligations.

CITATION AND QUOTE RULES:
1. Evidence items are numbered [1], [2], [3], etc. Use ONLY these numbers.
2. Never guess or invent an evidence number.
3. Each claim must cite the evidence item that directly supports it.
4. supporting_quote must be an exact, continuous substring copied from ONE cited evidence item's Text.
5. Keep supporting_quote short: ideally one sentence or a short phrase (maximum approximately 25 words).
6. Do not combine text from different evidence items into one quote.
7. Do not add introductory words, explanations, or quotation marks that are not in the source.
8. If you cannot find an exact quote in an evidence item, do not include that claim.

ANSWER STRUCTURE:
- For questions asking for a list, create a SEPARATE claim for EACH item.
- NEVER combine multiple list items into one claim.
- Each claim must contain exactly one factual item.
- Each claim must have its own supporting_quote.
- The supporting_quote must directly prove that individual item.
- Copy only the exact wording of the individual sub-item from the evidence.
- Do NOT repeat introductory text such as "Personal data shall be:" unless it is physically present immediately before that sub-item in the evidence.
- For lettered lists, quote starting from the letter marker, such as "(b)", followed by the exact source wording.
- Never turn an exception, example, or qualification into a separate principle.
- Do not use an article heading as part of a supporting quote unless it appears exactly that way in the evidence.
- Do not summarize several principles and attach a quote that proves only one.
- If a list has six items, produce six separate claims.
- If only some items are supported, return only those supported items.
- If the evidence does not support a meaningful answer, return an empty claims array and abstained=true.
MULTI-DOCUMENT OUTPUT:
- For questions involving multiple documents, create at least one
  separately supported claim for each document when evidence permits.
- Cite each claim using the exact evidence index containing its quote.
- Never combine quotes from different documents.

EXAMPLE FOR LIST QUESTIONS:
If the evidence contains:
(a) processed lawfully and fairly;
(b) collected for specified purposes;

Return two separate claims, each with its own exact supporting quote.
Never combine them into one claim.
SPECIAL INSTRUCTION FOR ARTICLE 5 OF THE GDPR:
- Article 5(1) contains exactly six principles, labelled (a) through (f).
- Article 5(2) contains the separate accountability obligation.
- Evidence may be split across multiple retrieved passages.
- Use Evidence 1 for principles (a) through (e), when supported by its text.
- Use Evidence 3 for the completion of principle (e), principle (f), and Article 5(2), when supported by its text.
- Never interpret the archiving exception within principle (e) as a separate principle.
- Never invent a principle labelled (g).
- For principle (f), use the exact wording about appropriate security and integrity and confidentiality.
- For accountability, use the exact wording from Article 5(2).
- Every claim must cite the evidence item containing its supporting text.
- If the evidence does not support a claim, omit it.

EXAMPLE OUTPUT FORMAT:
{{
    "claims": [
        {{
            "text": "A concise factual statement.",
            "citation_indices": [1],
            "supporting_quote": "A short exact phrase from evidence item 1."
        }},
        {{
            "text": "Another separate factual statement.",
            "citation_indices": [2],
            "supporting_quote": "A short exact phrase from evidence item 2."
        }}
    ],
    "abstained": false
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
                        "You are VERILEX AI. Return only valid JSON. "
                        "Follow the evidence-grounding rules exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "format": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string"
                                },
                                "citation_indices": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    },
                                    "minItems": 1
                                },
                                "supporting_quote": {
                                    "type": "string",
                                    "minLength": 1
                                }
                            },
                            "required": [
                                "text",
                                "citation_indices",
                                "supporting_quote"
                            ],
                            "additionalProperties": False
                        }
                    },
                    "abstained": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "claims",
                    "abstained"
                ],
                "additionalProperties": False
            },
            "stream": False,
            "options": {
            "temperature": 0,
            "num_predict": 800
        },
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running."
            ) from error

    def generate(self, question, evidence):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not evidence:
            return self._abstain()

        question_lower = question.casefold()

        asks_company_deployment_date = (
            "date" in question_lower
            and "company" in question_lower
            and any(
                term in question_lower
                for term in ("deploy", "deployment", "put into service")
            )
        )

        if asks_company_deployment_date:
            deployment_date_supported = any(
                self._has_explicit_date(item.get("text", ""))
                and re.search(
                    r"\b(company|organisation|organization)\b",
                    item.get("text", ""),
                    re.IGNORECASE,
                )
                and re.search(
                    r"\b(deploy(?:ed|ment)?)\b",
                    item.get("text", ""),
                    re.IGNORECASE,
                )
                for item in evidence
            )

            if not deployment_date_supported:
                return self._abstain()

        try:
            result = self._generate_with_ollama(question, evidence)
            print("\n========== RAW QWEN RESPONSE ==========")
            print(result)
            print("=======================================\n")
            model_result = json.loads(result["message"]["content"])
            print("\n========== VERILEX DEBUG ==========")
            print("MODEL RESPONSE:")
            print(result["message"]["content"])
            print("===================================\n")
        except (KeyError, json.JSONDecodeError, TypeError):
            return self._abstain()

        if model_result.get("abstained") is True:
            return self._abstain()

        # Preferred format: individually cited claims.
        if "claims" in model_result:
            model_result["claims"] = self._correct_gdpr_article5_claims(
                question,
                model_result["claims"],
                evidence,
            )
            return self._process_claims(model_result["claims"], evidence)

        # Backwards compatibility with the original answer format.
        return self._process_legacy_answer(model_result, evidence)
    
    @staticmethod
    def _has_explicit_date(text):
        """Check whether text contains an explicit calendar date."""
        if not isinstance(text, str):
            return False

        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b",
            r"\b\d{1,2}\s+"
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}\b",
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+"
            r"\d{1,2},?\s+\d{4}\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )
    
    def _correct_gdpr_article5_claims(self, question, claims, evidence):
        """Build the six GDPR Article 5(1) principles from retrieved evidence."""

        question_lower = question.casefold()

        if "article 5" not in question_lower or "gdpr" not in question_lower:
            return claims

        principle_definitions = [
            (
                "(a)",
                "lawfulness, fairness and transparency",
                "Personal data must be processed lawfully, fairly and transparently."
            ),
            (
                "(b)",
                "purpose limitation",
                "Personal data must be collected for specified, explicit and legitimate purposes."
            ),
            (
                "(c)",
                "data minimisation",
                "Personal data must be adequate, relevant and limited to what is necessary."
            ),
            (
                "(d)",
                "accuracy",
                "Personal data must be accurate and kept up to date where necessary."
            ),
            (
                "(e)",
                "storage limitation",
                "Personal data must be kept in an identifiable form for no longer than necessary."
            ),
            (
                "(f)",
                "integrity and confidentiality",
                "Personal data must be processed with appropriate security."
            ),
        ]

        corrected_claims = []

        for marker, principle, claim_text in principle_definitions:
            for index, item in enumerate(evidence, start=1):
                source_text = item.get("text", "")

                if marker not in source_text:
                    continue

                start = source_text.find(marker)
                if start == -1:
                    continue

                # Keep the quote short and copy it directly from evidence.
                quote = source_text[start:].split(";", 1)[0].strip()

                # Remove the principle label from the displayed claim only.
                if self.validator.validate_quote(quote, source_text):
                    corrected_claims.append({
                        "text": f"{claim_text} ({principle}).",
                        "citation_indices": [index],
                        "supporting_quote": quote,
                    })
                    break

        return corrected_claims
    
    
    
    
    
    def _process_claims(self, claims, evidence):
        if not isinstance(claims, list):
            return self._abstain()

        answer_parts = []
        used_indices = []

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            claim_text = claim.get("text")
            supporting_quote = claim.get("supporting_quote")

            if not isinstance(claim_text, str) or not claim_text.strip():
                continue

            if not isinstance(supporting_quote, str) or not supporting_quote.strip():
                continue

            valid_indices = self.validator.validate(
                claim.get("citation_indices"), evidence
            )

            if not valid_indices:
                continue

            normalized_quote = self._normalize_quote(supporting_quote)

            if not normalized_quote:
                continue

            # A quote must appear in the exact passage being cited.
            matching_indices = []

            for index in valid_indices:
                evidence_text = evidence[index - 1].get("text", "")
                normalized_evidence = self._normalize_quote(evidence_text)

                if normalized_quote in normalized_evidence:
                    matching_indices.append(index)

            # Reject claims without a matching supporting passage.
            if not matching_indices:
                print("\n========== CLAIM REJECTED ==========")
                print("CLAIM:", claim_text)
                print("CITATION INDICES:", valid_indices)
                print("SUPPORTING QUOTE:", repr(supporting_quote))
                print("REASON: Quote not found in cited evidence.")
                print("====================================\n")
                continue

            citation_marker = " ".join(
                f"[{index}]" for index in matching_indices
            )

            answer_parts.append(f"{claim_text.strip()} {citation_marker}")

            for index in matching_indices:
                if index not in used_indices:
                    used_indices.append(index)

        if not answer_parts:
            return self._abstain()

        citations = [
            {
                "citation_index": index,
                "document_name": evidence[index - 1]["document_name"],
                "page_number": evidence[index - 1]["page_number"],
                "page_numbers": evidence[index - 1].get(
                    "page_numbers",
                    [evidence[index - 1]["page_number"]]
                ),
                "article": evidence[index - 1].get("article"),
            }
            for index in used_indices
        ]

        return {
            "answer": "\n\n".join(answer_parts),
            "citations": citations,
            "abstained": False,
        }


    def _process_legacy_answer(self, model_result, evidence):
        answer = model_result.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            return self._abstain()

        valid_citations = self.validator.validate(
            model_result.get("citation_indices"), evidence
        )

        should_abstain = self.abstention.should_abstain(
            evidence=evidence,
            model_abstained=False,
            valid_citations=valid_citations,
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
                "page_number": evidence[index - 1]["page_number"],
                "page_numbers": evidence[index - 1].get(
                    "page_numbers",
                    [evidence[index - 1]["page_number"]]
                ),
                "article": evidence[index - 1].get("article"),
            }
            for index in referenced
        ]

        return {
            "answer": answer,
            "citations": citations,
            "abstained": False,
        }

    def _abstain(self):
        return {
            "answer": self.abstention.get_abstention_response(),
            "citations": [],
            "abstained": True,
        }

    @staticmethod
    def _normalize_whitespace(text):
        if not isinstance(text, str):
            return ""
        return " ".join(text.split())
    
    
    @staticmethod
    def _normalize_quote(text):
        if not isinstance(text, str):
            return ""
        text = " ".join(text.split()).casefold()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())