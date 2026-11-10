
from io import BytesIO
from pathlib import Path
import tempfile

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from ingestion.pdf_loader import PDFLoader


def test_scanned_pdf_ocr():
    # Create an image containing text.
    image = Image.new("RGB", (1600, 500), "white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf", 55
    )

    draw.text(
        (60, 100),
        "Article 5 - Principles of data processing",
        fill="black",
        font=font
    )

    draw.text(
        (60, 220),
        "Personal data shall be processed lawfully and fairly.",
        fill="black",
        font=font
    )

    # Convert the image into a PDF page.
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")

    pdf = pymupdf.open()
    page = pdf.new_page(width=800, height=250)
    page.insert_image(
        page.rect,
        stream=image_buffer.getvalue()
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "scanned_test.pdf"
        pdf.save(pdf_path)
        pdf.close()

        # Load the scanned PDF through VERILEX AI.
        result = PDFLoader().load(pdf_path)

        scanned_page = result["pages"][0]

        print("OCR used:", scanned_page["ocr_used"])
        print("Extracted text:", scanned_page["text"])

        assert scanned_page["is_scanned"] is True
        assert scanned_page["ocr_used"] is True
        assert "Article 5" in scanned_page["text"]
        assert "lawfully" in scanned_page["text"]

    print("\nScanned PDF OCR test passed!")


if __name__ == "__main__":
    test_scanned_pdf_ocr()
