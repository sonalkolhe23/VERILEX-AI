
from io import BytesIO
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


class OCRProcessor:
    """Extract text from scanned PDF pages using Tesseract OCR."""

    def __init__(self, language="eng", zoom=2):
        self.language = language
        self.zoom = zoom

        # Windows Tesseract installation
        tesseract_path = Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        if tesseract_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(
                tesseract_path
            )

    def extract_text(self, page):
        """Render a PDF page and extract its text using OCR."""

        matrix = pymupdf.Matrix(self.zoom, self.zoom)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image_bytes = pixmap.tobytes("png")

        image = Image.open(BytesIO(image_bytes))

        text = pytesseract.image_to_string(
            image,
            lang=self.language,
            config="--psm 3"
        )

        return text.strip()
