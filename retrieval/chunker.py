
import re
import hashlib
from typing import Optional


class LegalChunker:
    """
    Splits legal documents into searchable chunks while preserving
    document identity, legal structure and page-level references.
    """

    CHAPTER_PATTERN = re.compile(
        r"^\s*CHAPTER\s+([IVXLCDM]+)\s*$",
        re.IGNORECASE
    )

    SECTION_PATTERN = re.compile(
        r"^\s*SECTION\s+(\d+)\s*$",
        re.IGNORECASE
    )

    ARTICLE_PATTERN = re.compile(
        r"^\s*Article\s+(\d+)\s*$",
        re.IGNORECASE
    )

    def __init__(
        self,
        max_chars: int = 1800,
        overlap_chars: int = 200,
        min_chars: int = 100
    ):
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")

        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError(
                "overlap_chars must be non-negative and smaller "
                "than max_chars."
            )

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    @staticmethod
    def _make_chunk_id(
        document_id: str,
        page_number: int,
        chunk_index: int,
        text: str
    ) -> str:
        content = (
            f"{document_id}|{page_number}|"
            f"{chunk_index}|{text}"
        )

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:16]

    def _split_text(self, text: str) -> list[str]:
        """
        Splits long text at paragraph or sentence boundaries
        where possible, with controlled character overlap.
        """

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.max_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chars, len(text))

            if end < len(text):
                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("; ", start, end)
                )

                if boundary > start + self.max_chars // 2:
                    end = boundary + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            next_start = max(
                start + 1,
                end - self.overlap_chars
            )

            start = next_start

        return chunks

    def chunk_pages(
        self,
        pages: list[dict],
        document_id: Optional[str] = None,
        document_name: str = "Unknown document"
    ) -> list[dict]:
        """
        Creates page-specific chunks and preserves the legal
        hierarchy detected in the document.

        Expected page format:
        {
            "page_number": 1,
            "text": "..."
        }
        """

        if not pages:
            return []

        if document_id is None:
            document_id = hashlib.sha256(
                document_name.encode("utf-8")
            ).hexdigest()[:12]

        current_chapter = None
        current_section = None
        current_article = None

        chunks = []
        chunk_index = 0

        for page in pages:
            page_number = page["page_number"]
            page_text = page.get("text", "")

            if not page_text.strip():
                continue

            lines = page_text.splitlines()
            content_lines = []

            def flush_content():
                nonlocal chunk_index

                content = "\n".join(content_lines).strip()

                if not content:
                    return

                for part in self._split_text(content):
                    if len(part) < self.min_chars and chunks:
                        # Keep short fragments rather than discarding
                        # potentially important legal text.
                        pass

                    chunk_id = self._make_chunk_id(
                        document_id,
                        page_number,
                        chunk_index,
                        part
                    )

                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_number,
                        "page_numbers": [page_number],
                        "chapter": current_chapter,
                        "section": current_section,
                        "article": current_article,
                        "text": part,
                        "character_count": len(part)
                    })

                    chunk_index += 1

                content_lines.clear()

            for line in lines:
                stripped = line.strip()

                chapter_match = self.CHAPTER_PATTERN.match(
                    stripped
                )

                section_match = self.SECTION_PATTERN.match(
                    stripped
                )

                article_match = self.ARTICLE_PATTERN.match(
                    stripped
                )

                if chapter_match:
                    flush_content()
                    current_chapter = (
                        f"Chapter {chapter_match.group(1).upper()}"
                    )
                    current_section = None
                    current_article = None

                    content_lines.append(stripped)
                    continue

                if section_match:
                    flush_content()
                    current_section = (
                        f"Section {section_match.group(1)}"
                    )
                    current_article = None

                    content_lines.append(stripped)
                    continue

                if article_match:
                    flush_content()
                    current_article = (
                        f"Article {article_match.group(1)}"
                    )

                    content_lines.append(stripped)
                    continue

                content_lines.append(line)

            flush_content()

        return chunks

    def chunk_document(
        self,
        document: dict,
        document_id: Optional[str] = None
    ) -> list[dict]:
        """
        Accepts the output of PDFLoader directly.
        """

        return self.chunk_pages(
            pages=document["pages"],
            document_id=document_id,
            document_name=document.get(
                "document_name",
                "Unknown document"
            )
        )
