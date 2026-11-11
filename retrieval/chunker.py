
import re
import hashlib
from typing import Optional


class LegalChunker:
    """
    Splits legal documents into searchable chunks while preserving
    legal article boundaries and accurate page-level references.
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
                "overlap_chars must be non-negative and "
                "smaller than max_chars."
            )

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    @staticmethod
    def _make_chunk_id(
        document_id: str,
        chunk_index: int,
        text: str
    ) -> str:
        content = f"{document_id}|{chunk_index}|{text}"
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:16]


    def _split_with_pages(
        self,
        lines: list[tuple[str, int]]
    ) -> list[tuple[str, list[int]]]:
        """Split text while retaining accurate page references."""

        # Remove empty lines and trim each line before building
        # the character-to-page mapping.
        clean_lines = [
            (line.strip(), page_number)
            for line, page_number in lines
            if line.strip()
        ]

        if not clean_lines:
            return []

        text_parts = []
        char_pages = []

        for index, (line, page_number) in enumerate(clean_lines):
            if index > 0:
                text_parts.append("\n")
                char_pages.append(page_number)

            text_parts.append(line)
            char_pages.extend([page_number] * len(line))

        text = "".join(text_parts)

        results = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chars, len(text))

            if end < len(text):
                boundaries = [
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("; ", start, end)
                ]

                boundary = max(boundaries)

                if boundary > start + self.max_chars // 2:
                    end = boundary + 1

            part = text[start:end].strip()

            if part:
                page_numbers = sorted(set(char_pages[start:end]))
                results.append((part, page_numbers))

            if end >= len(text):
                break

            start = max(start + 1, end - self.overlap_chars)

        return results


    def chunk_pages(
        self,
        pages: list[dict],
        document_id: Optional[str] = None,
        document_name: str = "Unknown document"
    ) -> list[dict]:

        if not pages:
            return []

        if document_id is None:
            document_id = hashlib.sha256(
                document_name.encode("utf-8")
            ).hexdigest()[:12]

        current_chapter = None
        current_section = None
        current_article = None

        article_lines: list[tuple[str, int]] = []
        chunks = []
        chunk_index = 0

        def flush_article():
            nonlocal chunk_index

            if not article_lines:
                return

            for part, page_numbers in self._split_with_pages(
                article_lines
            ):
                chunks.append({
                    "chunk_id": self._make_chunk_id(
                        document_id,
                        chunk_index,
                        part
                    ),
                    "document_id": document_id,
                    "document_name": document_name,
                    "page_number": page_numbers[0],
                    "page_numbers": page_numbers,
                    "chapter": current_chapter,
                    "section": current_section,
                    "article": current_article,
                    "text": part,
                    "character_count": len(part)
                })
                chunk_index += 1

            article_lines.clear()

        for page in pages:
            page_number = page["page_number"]
            page_text = page.get("text", "")

            if not page_text.strip():
                continue

            for line in page_text.splitlines():
                stripped = line.strip()

                if not stripped:
                    continue

                chapter_match = self.CHAPTER_PATTERN.match(stripped)
                section_match = self.SECTION_PATTERN.match(stripped)
                article_match = self.ARTICLE_PATTERN.match(stripped)

                if chapter_match:
                    flush_article()
                    current_chapter = (
                        f"Chapter {chapter_match.group(1).upper()}"
                    )
                    current_section = None
                    current_article = None

                elif section_match:
                    flush_article()
                    current_section = (
                        f"Section {section_match.group(1)}"
                    )
                    current_article = None

                elif article_match:
                    flush_article()
                    current_article = (
                        f"Article {article_match.group(1)}"
                    )

                article_lines.append((line, page_number))

        flush_article()
        return chunks

    def chunk_document(
        self,
        document: dict,
        document_id: Optional[str] = None
    ) -> list[dict]:
        """Accept the output of PDFLoader directly."""

        return self.chunk_pages(
            pages=document["pages"],
            document_id=document_id,
            document_name=document.get(
                "document_name",
                "Unknown document"
            )
        )
