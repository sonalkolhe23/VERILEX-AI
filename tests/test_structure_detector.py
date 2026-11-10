
from ingestion.pdf_loader import PDFLoader
from ingestion.structure_detector import StructureDetector


def test_gdpr_structure_detection():
    loader = PDFLoader()
    detector = StructureDetector()

    document = loader.load(
        "data/uploads/sample_legal_document.pdf"
    )

    result = detector.detect(document["pages"])

    chapters = {
        heading["heading"]
        for heading in result["headings"]
        if heading["type"] == "chapter"
    }

    articles = {
        int(heading["number"])
        for heading in result["headings"]
        if heading["type"] == "article"
    }

    sections = [
        heading
        for heading in result["headings"]
        if heading["type"] == "section"
    ]

    print("Total pages:", len(result["pages"]))
    print("Chapters detected:", len(chapters))
    print("Chapter names:", sorted(chapters))
    print("Unique articles detected:", len(articles))
    print("Article numbers:", sorted(articles))
    print("Section headings detected:", len(sections))

    assert len(result["pages"]) == 78
    assert len(chapters) == 11
    assert articles == set(range(1, 100))

    print("\nGDPR structure detection test passed!")


if __name__ == "__main__":
    test_gdpr_structure_detection()
