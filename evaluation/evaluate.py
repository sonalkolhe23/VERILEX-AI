import json
import time
from pathlib import Path

from ingestion.document_manager import DocumentManager
from generation.answer_generator import AnswerGenerator


ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "data" / "uploads"
QUESTIONS_FILE = ROOT / "evaluation" / "questions.json"
RESULTS_FILE = ROOT / "evaluation" / "results.json"
SUMMARY_FILE = ROOT / "evaluation" / "summary.json"

PDF_FILES = [
    UPLOADS / "eu_ai_act.pdf",
    UPLOADS / "sample_legal_document.pdf",
]

TOP_K = 5


def load_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def page_refs(results):
    """Return retrieved (document name, page number) pairs."""
    refs = set()

    for chunk in results:
        document = chunk.get("document_name", "")
        pages = chunk.get("page_numbers") or [chunk.get("page_number")]

        for page in pages:
            if page is not None:
                refs.add((document, int(page)))

    return refs


def expected_refs(question):
    """Return the annotated evidence document/page pairs."""
    refs = set()

    for item in question.get("evidence", []):
        document = item["document"]

        for page in item.get("pages", []):
            refs.add((document, int(page)))

    return refs


def calculate_recall_at_k(question, retrieved):
    relevant = expected_refs(question)

    if not relevant:
        return None

    found = page_refs(retrieved)
    return len(relevant.intersection(found)) / len(relevant)


def select_document_balanced(results, top_k=5, max_per_document=3):
    """
    Select relevant chunks while preventing one document
    from dominating the final evidence set.
    """
    selected = []
    document_counts = {}

    for result in sorted(
        results,
        key=lambda item: item.get("rerank_score", 0),
        reverse=True
    ):
        document = result.get("document_name", "Unknown")

        if document_counts.get(document, 0) >= max_per_document:
            continue

        selected.append(result)
        document_counts[document] = (
            document_counts.get(document, 0) + 1
        )

        if len(selected) >= top_k:
            break

    return selected


def citation_precision(citations, question):
    """Measure how many generated citations match annotated evidence pages."""
    if not citations:
        return None

    relevant = expected_refs(question)
    supported = 0
    total = 0

    for citation in citations:
        document = citation.get("document_name", "")
        pages = citation.get("page_numbers") or [citation.get("page_number")]

        for page in pages:
            if page is None:
                continue

            total += 1
            if (document, int(page)) in relevant:
                supported += 1

    return supported / total if total else None


def safe_average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main():
    questions = load_questions()

    for pdf in PDF_FILES:
        if not pdf.exists():
            raise FileNotFoundError(f"Required PDF not found: {pdf}")

    print("Loading legal documents...")
    manager = DocumentManager()

    for pdf in PDF_FILES:
        result = manager.add_document(str(pdf))
        print(
            f"Indexed: {result['document_name']} | "
            f"Pages: {result['total_pages']} | "
            f"Chunks: {result['total_chunks']}"
        )

    print("\nBuilding retrieval index...")
    manager.build_index()

    generator = AnswerGenerator()
    results = []

    for number, question_data in enumerate(questions, start=1):
        question = question_data["question"]
        print(f"\n[{number}/{len(questions)}] {question_data['id']}: {question}")

        # Dense-only retrieval
        start = time.perf_counter()
        dense_results = manager.search_engine.dense.search(
            question, top_k=TOP_K
        )
        dense_latency = time.perf_counter() - start

        # Hybrid retrieval
        start = time.perf_counter()
        hybrid_results = manager.search(question, top_k=TOP_K)
        hybrid_latency = time.perf_counter() - start

        dense_recall = calculate_recall_at_k(question_data, dense_results)
        hybrid_recall = calculate_recall_at_k(question_data, hybrid_results)
        
        # Expanded evidence pool for document-balanced answer generation
        expanded_results = manager.search(question, top_k=20)

        answer_evidence = select_document_balanced(
            expanded_results,
            top_k=5,
            max_per_document=3
        )
        
        # Generate an answer using the hybrid evidence.
        start = time.perf_counter()
        try:
            generated = generator.generate(question, answer_evidence)
            generation_error = None
        except Exception as error:
            generated = {
                "answer": "",
                "citations": [],
                "abstained": False,
            }
            generation_error = str(error)

        generation_latency = time.perf_counter() - start

        expected_abstention = question_data["should_abstain"]
        actual_abstention = bool(generated.get("abstained", False))

        citations = generated.get("citations", [])
        precision = citation_precision(citations, question_data)

        record = {
            "id": question_data["id"],
            "type": question_data["type"],
            "question": question,
            "expected_answer": question_data.get("expected_answer"),
            "answer": generated.get("answer", ""),
            "citations": citations,
            "expected_abstention": expected_abstention,
            "actual_abstention": actual_abstention,
            "abstention_correct": (
                expected_abstention == actual_abstention
            ),
            "dense_recall_at_5": dense_recall,
            "hybrid_recall_at_5": hybrid_recall,
            "citation_precision": precision,
            "dense_retrieval_seconds": round(dense_latency, 3),
            "hybrid_retrieval_seconds": round(hybrid_latency, 3),
            "generation_seconds": round(generation_latency, 3),
            "generation_error": generation_error,
            "manual_answer_review_required": (
                question_data["type"] in ("factual", "multi_hop")
                and not actual_abstention
            ),
        }

        results.append(record)

        print(
            f"  Dense Recall@5: {dense_recall} | "
            f"Hybrid Recall@5: {hybrid_recall}"
        )
        print(
            f"  Abstained: {actual_abstention} | "
            f"Generation: {generation_latency:.2f}s"
        )

    # Overall metrics
    factual_and_multihop = [
        item for item in results
        if item["type"] in ("factual", "multi_hop")
    ]
    unanswerable = [
        item for item in results
        if item["type"] == "unanswerable"
    ]

    summary = {
        "total_questions": len(results),
        "question_breakdown": {
            kind: sum(item["type"] == kind for item in results)
            for kind in ("factual", "multi_hop", "unanswerable")
        },
        "dense_only": {
            "mean_recall_at_5": safe_average(
                [item["dense_recall_at_5"] for item in results]
            )
        },
        "hybrid": {
            "mean_recall_at_5": safe_average(
                [item["hybrid_recall_at_5"] for item in results]
            )
        },
        "mean_citation_precision": safe_average(
            [item["citation_precision"] for item in factual_and_multihop]
        ),
        "abstention_accuracy": safe_average(
            [
                float(item["abstention_correct"])
                for item in unanswerable
            ]
        ),
        "mean_dense_retrieval_seconds": safe_average(
            [item["dense_retrieval_seconds"] for item in results]
        ),
        "mean_hybrid_retrieval_seconds": safe_average(
            [item["hybrid_retrieval_seconds"] for item in results]
        ),
        "mean_generation_seconds": safe_average(
            [item["generation_seconds"] for item in results]
        ),
        "answer_correctness": (
            "Requires manual review against expected_answer."
        ),
        "notes": [
            "Recall@5 is calculated using annotated document/page pairs.",
            "Citation precision checks generated citation pages against annotated evidence.",
            "Unanswerable questions are evaluated using abstention accuracy.",
            "Factual and multi-hop answer correctness requires manual review."
        ]
    }

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
