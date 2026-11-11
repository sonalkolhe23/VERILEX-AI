
import hashlib
from pathlib import Path

from ingestion.pdf_loader import PDFLoader
from ingestion.structure_detector import StructureDetector
from retrieval.chunker import LegalChunker
from retrieval.hybrid_search import HybridSearch


class DocumentManager:
    """Manages multiple legal PDFs and their searchable chunks."""

    def __init__(
        self,
        pdf_loader=None,
        structure_detector=None,
        chunker=None,
        search_engine=None
    ):
        self.pdf_loader = pdf_loader or PDFLoader()
        self.structure_detector = (
            structure_detector or StructureDetector()
        )
        self.chunker = chunker or LegalChunker()
        self.search_engine = search_engine

        self.documents = {}
        self.chunks = []
        self.is_indexed = False

    @staticmethod
    def _generate_document_id(pdf_path):
        """Generate a stable ID from the PDF file contents."""
        hasher = hashlib.sha256()

        with open(pdf_path, "rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024), b""
            ):
                hasher.update(block)

        return hasher.hexdigest()[:16]

    def add_document(self, pdf_path, document_name=None):
        """Load one PDF and add its chunks to the collection."""
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        document_id = self._generate_document_id(pdf_path)

        if document_id in self.documents:
            raise ValueError(
                f"Document already uploaded: {pdf_path.name}"
            )

        document = self.pdf_loader.load(pdf_path)
        
        if document_name:
            document["document_name"] = Path(document_name).name

        structure = self.structure_detector.detect(
            document["pages"]
        )

        document["pages"] = structure["pages"]
        document["headings"] = structure["headings"]
        document["document_id"] = document_id

        document_chunks = self.chunker.chunk_document(
            document,
            document_id=document_id
        )

        self.documents[document_id] = document
        self.chunks.extend(document_chunks)

        # The index must be rebuilt after adding documents.
        self.is_indexed = False

        return {
            "document_id": document_id,
            "document_name": document["document_name"],
            "total_pages": document["total_pages"],
            "total_chunks": len(document_chunks),
            "total_headings": len(document["headings"])
        }

    def add_documents(self, pdf_paths):
        """Add multiple PDFs to the collection."""
        results = []

        for pdf_path in pdf_paths:
            results.append(self.add_document(pdf_path))

        return results

    def build_index(self):
        """Build one hybrid index across all uploaded documents."""
        if not self.chunks:
            raise ValueError(
                "No document chunks available for indexing."
            )

        if self.search_engine is None:
            self.search_engine = HybridSearch()

        self.search_engine.fit(self.chunks)
        self.is_indexed = True

        return len(self.chunks)

    def search(self, question, top_k=5):
        """Search across all uploaded legal documents."""
        if not self.is_indexed:
            raise RuntimeError(
                "Build the document index before searching."
            )

        return self.search_engine.search(
            question,
            top_k=top_k
        )

    def get_documents(self):
        """Return metadata for all uploaded documents."""
        return [
            {
                "document_id": document_id,
                "document_name": document["document_name"],
                "total_pages": document["total_pages"],
                "total_headings": len(document["headings"])
            }
            for document_id, document in self.documents.items()
        ]

    def get_document(self, document_id):
        """Retrieve one document using its unique ID."""
        return self.documents.get(document_id)
