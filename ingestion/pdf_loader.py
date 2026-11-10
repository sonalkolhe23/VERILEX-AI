
from pathlib import Path
import pymupdf


class PDFLoader:
    """
    Extracts text and metadata from legal PDF documents.
    Preserves page-level information for evidence citations.
    """

    def __init__(self, min_text_length=30):
        self.min_text_length = min_text_length

    def load(self, pdf_path):
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        pages = []

        try:
            with pymupdf.open(pdf_path) as document:

                if document.needs_pass:
                    raise ValueError(
                        "Password-protected PDFs are not supported."
                    )

                metadata = document.metadata or {}

                for page_number, page in enumerate(document, start=1):

                    text = page.get_text("text", sort=True).strip()

                    is_scanned = (
                        len(text) < self.min_text_length
                    )

                    pages.append({
                        "page_number": page_number,
                        "text": text,
                        "character_count": len(text),
                        "is_scanned": is_scanned,
                        "width": round(page.rect.width, 2),
                        "height": round(page.rect.height, 2)
                    })

                return {
                    "document_name": pdf_path.name,
                    "document_path": str(pdf_path.resolve()),
                    "total_pages": len(pages),
                    "metadata": {
                        "title": metadata.get("title"),
                        "author": metadata.get("author"),
                        "subject": metadata.get("subject"),
                        "creator": metadata.get("creator")
                    },
                    "pages": pages
                }

        except pymupdf.FileDataError as error:
            raise ValueError(
                f"Invalid or corrupted PDF: {error}"
            ) from error


if __name__ == "__main__":
    print("VERILEX AI PDF Loader")
    print("Import PDFLoader to process a document.")
