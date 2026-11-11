
import json
import urllib.request
import urllib.error

from collections import defaultdict


class DocumentComparator:
    """Compares legal evidence across multiple uploaded documents."""

    def __init__(
        self,
        document_manager,
        model="qwen2.5:3b",
        base_url="http://localhost:11434"
    ):
        self.document_manager = document_manager
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _retrieve_evidence(
        self,
        topic,
        document_ids=None,
        top_k_per_document=3
    ):
        """Retrieve and group relevant evidence by document."""

        documents = self.document_manager.get_documents()

        if document_ids is not None:
            requested_ids = set(document_ids)
            known_ids = {
                document["document_id"] for document in documents
            }

            unknown_ids = requested_ids - known_ids

            if unknown_ids:
                raise ValueError(
                    f"Unknown document IDs: {sorted(unknown_ids)}"
                )

            documents = [
                document for document in documents
                if document["document_id"] in requested_ids
            ]

        if len(documents) < 2:
            return []

        search_results = self.document_manager.search(
            topic,
            top_k=max(10, len(documents) * top_k_per_document * 3)
        )

        allowed_ids = {
            document["document_id"] for document in documents
        }

        grouped = defaultdict(list)
        seen = set()

        for result in search_results:
            document_id = result.get("document_id")

            if document_id not in allowed_ids:
                continue

            text = result.get("text", "").strip()

            # Ignore empty or heading-only retrieval results.
            if len(text) < 100:
                continue

            key = (
                document_id,
                result.get("page_number"),
                text
            )

            if key in seen:
                continue

            seen.add(key)
            grouped[document_id].append(result)

        evidence = []

        for document in documents:
            document_id = document["document_id"]
            matches = grouped.get(document_id, [])

            for result in matches[:top_k_per_document]:
                evidence.append({
                    "document_id": document_id,
                    "document_name": result.get(
                        "document_name",
                        document["document_name"]
                    ),
                    "page_number": result.get("page_number"),
                    "article": result.get("article"),
                    "text": result.get("text", "")
                })

        represented_documents = {
            item["document_id"] for item in evidence
        }

        if len(represented_documents) < 2:
            return []

        return evidence

    def _generate_with_ollama(self, topic, evidence):
        """Ask Ollama to compare only the retrieved evidence."""

        evidence_text = ""

        for index, item in enumerate(evidence, start=1):
            evidence_text += (
                f"\n[Evidence {index}]\n"
                f"Document: {item['document_name']}\n"
                f"Page: {item['page_number']}\n"
                f"Article: {item.get('article')}\n"
                f"Text: {item['text']}\n"
            )

        prompt = f"""
You are VERILEX AI, an evidence-grounded legal document analyst.

Compare the supplied legal documents about this topic:
{topic}

Use ONLY the supplied evidence.

Rules:
1. Never use outside legal knowledge.
2. Never invent legal requirements or conclusions.
3. Distinguish similarities from differences.
4. A difference is NOT automatically a legal conflict.
5. Report a potential conflict only when the cited
   requirements appear incompatible.
6. If the evidence does not establish incompatibility,
   do not call it a conflict.
7. Every comparison item must cite evidence from at least
   two different documents.
8. Use only valid evidence numbers.
9. If evidence is insufficient, return empty arrays.
10. Return valid JSON only.

Use exactly this structure:
{{
    "summary": "Short comparison overview",
    "similarities": [
        {{
            "claim": "Shared point",
            "citation_indices": [1, 2]
        }}
    ],
    "differences": [
        {{
            "claim": "Difference between the documents",
            "citation_indices": [1, 2]
        }}
    ],
    "potential_conflicts": [
        {{
            "claim": "Potential incompatibility",
            "reason": "Why the requirements may conflict",
            "citation_indices": [1, 2]
        }}
    ],
    "overall_assessment": "Careful overall assessment"
}}

EVIDENCE:
{evidence_text}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You compare legal documents using only "
                        "provided evidence. Return valid JSON."
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
            with urllib.request.urlopen(
                request,
                timeout=180
            ) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )

        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ) from error

    
    @staticmethod
    def _is_non_conflict(item):
        """Detect claims that explicitly say no conflict exists."""

        text = (
            str(item.get("claim", "")) + " " +
            str(item.get("reason", ""))
        ).lower()

        phrases = [
            "no evidence of incompatibility",
            "no incompatibility",
            "no conflict",
            "not a conflict",
            "does not conflict",
            "do not conflict",
            "not incompatible",
            "compatible with",
            "there is no evidence"
        ]

        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _validate_items(items, evidence):
        """Validate citations and remove duplicate source references."""

        validated = []

        if not isinstance(items, list):
            return validated

        for item in items:
            if not isinstance(item, dict):
                continue

            indices = item.get("citation_indices")

            if not isinstance(indices, list):
                continue

            valid_indices = []

            for index in indices:
                if (
                    isinstance(index, int)
                    and not isinstance(index, bool)
                    and 1 <= index <= len(evidence)
                    and index not in valid_indices
                ):
                    valid_indices.append(index)

            cited_documents = {
                evidence[index - 1]["document_id"]
                for index in valid_indices
            }

            if len(cited_documents) < 2:
                continue

            claim = item.get("claim")

            if not isinstance(claim, str) or not claim.strip():
                continue

            citations = []
            seen_sources = set()

            for index in valid_indices:
                source = evidence[index - 1]

                citation_key = (
                    source["document_id"],
                    source.get("page_number"),
                    source.get("article")
                )

                if citation_key in seen_sources:
                    continue

                seen_sources.add(citation_key)

                citations.append({
                    "document_name": source["document_name"],
                    "page_number": source["page_number"],
                    "article": source.get("article")
                })

            validated_item = {
                "claim": claim,
                "citations": citations
            }

            if "reason" in item:
                validated_item["reason"] = item["reason"]

            validated.append(validated_item)

        return validated


    def compare(
        self,
        topic,
        document_ids=None,
        top_k_per_document=3
    ):
        """Compare multiple documents using retrieved evidence."""

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Comparison topic cannot be empty.")

        if top_k_per_document < 1:
            raise ValueError(
                "top_k_per_document must be at least 1."
            )

        evidence = self._retrieve_evidence(
            topic,
            document_ids=document_ids,
            top_k_per_document=top_k_per_document
        )

        if not evidence:
            return self._abstain(
                "Insufficient evidence from at least two "
                "documents to perform a comparison."
            )

        try:
            response = self._generate_with_ollama(
                topic,
                evidence
            )

            result = json.loads(
                response["message"]["content"]
            )

        except (
            KeyError,
            json.JSONDecodeError,
            TypeError
        ):
            return self._abstain(
                "The comparison could not be validated."
            )

        return {
            "topic": topic,
            "summary": result.get("summary", ""),
            "similarities": self._validate_items(
                result.get("similarities"),
                evidence
            ),
            "differences": self._validate_items(
                result.get("differences"),
                evidence
            ),
            "potential_conflicts": [
                item for item in self._validate_items(
                    result.get("potential_conflicts"),
                    evidence
                )
                if not self._is_non_conflict(item)
            ],
            "overall_assessment": result.get(
                "overall_assessment", ""
            ),
            "evidence_count": len(evidence),
            "abstained": False
        }

    @staticmethod
    def _abstain(message):
        return {
            "summary": message,
            "similarities": [],
            "differences": [],
            "potential_conflicts": [],
            "overall_assessment": message,
            "evidence_count": 0,
            "abstained": True
        }

