# VERILEX AI
### Evidence-Grounded Legal Document Analyst

VERILEX AI is an AI-powered legal document analysis prototype designed to help users explore legal documents, retrieve relevant evidence, and generate answers grounded in the source material.

## Key Features

- **PDF Ingestion:** Extracts text and document content from PDF files.
- **OCR Support:** Uses Tesseract OCR to process scanned documents.
- **Legal Structure Detection:** Identifies document structure and creates meaningful legal text chunks.
- **Hybrid Retrieval:** Combines semantic search and BM25 keyword retrieval.
- **Cross-Encoder Reranking:** Improves the relevance of retrieved evidence.
- **Evidence-Grounded Answers:** Generates answers supported by retrieved document passages.
- **Citation Validation:** Checks supporting evidence and helps prevent unsupported answers.
- **Abstention:** Can refrain from answering when sufficient evidence is unavailable.
- **Multi-Document Analysis:** Supports comparison and analysis across documents.
- **Evaluation Framework:** Measures retrieval performance, citation precision, and abstention accuracy.

## Technology Stack

- Python 3.11
- Streamlit
- PyMuPDF
- Tesseract OCR
- Sentence Transformers
- BM25
- Cross-Encoder Reranking
- Docker
- Pytest

## Project Structure

```text
VERILEX-AI/
├── analysis/
├── assets/
├── data/
├── database/
├── evaluation/
├── generation/
├── ingestion/
├── retrieval/
├── screenshots/
├── tests/
├── app.py
├── config.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/sonalkolhe23/VERILEX-AI.git
cd VERILEX-AI
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Run with Docker

Build the image:

```bash
docker build -t verilex-ai .
```

Start the application:

```bash
docker run --rm -p 8501:8501 verilex-ai
```

Open `http://localhost:8501`.

The Docker image includes the Tesseract system dependency required for OCR.

## Experimental Evaluation

The project includes a 20-question evaluation set:

- 10 factual questions
- 5 multi-hop questions
- 5 unanswerable questions

Initial experimental results:

| Metric | Result |
|---|---:|
| Dense Retrieval Recall@5 | 50.00% |
| Hybrid Retrieval Recall@5 | 68.89% |
| Citation Precision | 45.00% |
| Abstention Accuracy | 80.00% |

Hybrid retrieval improved Recall@5 by approximately 18.89 percentage points compared with dense retrieval.

These results are preliminary and based on a small evaluation set. Further testing and improvements are needed, particularly for citation precision and answer quality.

Detailed evaluation outputs are available in the `evaluation/` directory.

## Screenshots

Application screenshots and evaluation examples are available in the `screenshots/` directory.

## Limitations

VERILEX AI is an academic prototype. Its outputs should be independently verified against the original legal documents. It is not a substitute for professional legal advice.

## Repository

[VERILEX AI on GitHub](https://github.com/sonalkolhe23/VERILEX-AI)

**Project:** VERILEX AI  
**Purpose:** Evidence-grounded legal document analysis  
**Environment:** Python, Streamlit and Docker