
from ingestion.pdf_loader import PDFLoader
import re

loader = PDFLoader()

document = loader.load(
    "data/uploads/sample_legal_document.pdf"
)

for page in document["pages"]:
    for line in page["text"].splitlines():
        line = line.strip()

        if re.search(r"chapter|section|article", line, re.IGNORECASE):
            print(f"PAGE {page['page_number']}: {line!r}")
