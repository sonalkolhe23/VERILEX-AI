
from ingestion.pdf_loader import PDFLoader


def test_loader_initialization():
    loader = PDFLoader()

    assert loader.min_text_length == 30


def test_missing_file():
    loader = PDFLoader()

    try:
        loader.load("nonexistent.pdf")
        assert False, "Expected FileNotFoundError"

    except FileNotFoundError:
        assert True


if __name__ == "__main__":
    test_loader_initialization()
    test_missing_file()

    print("All PDF loader tests passed!")
