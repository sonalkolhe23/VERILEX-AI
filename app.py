
import streamlit as st
import tempfile
import time
from pathlib import Path

from ingestion.document_manager import DocumentManager
from generation.answer_generator import AnswerGenerator
from analysis.document_comparator import DocumentComparator


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VERILEX AI | Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# PREMIUM UI STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F5F6F8;
    color: #182230;
}

[data-testid="stSidebar"] {
    background: #111C2E;
}

[data-testid="stSidebar"] * {
    color: #F5F7FA;
}

.brand {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 1px;
}

.brand-subtitle {
    color: #A9B7C9;
    font-size: 10px;
    letter-spacing: 2px;
}

.hero {
    background: linear-gradient(120deg, #14243A, #263D5C);
    padding: 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 34px;
    color: white;
}

.hero p {
    color: #D6DFEA;
    font-size: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E5E9EF;
}

.metric-label {
    color: #65758B;
    font-size: 12px;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
    color: #192C45;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    color: #192C45;
}

.evidence-card {
    background: white;
    border: 1px solid #E1E6ED;
    border-left: 4px solid #54769C;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

.stTextInput input, .stTextArea textarea {
    border-radius: 8px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "document_manager" not in st.session_state:
    st.session_state.document_manager = DocumentManager()

if "answer_generator" not in st.session_state:
    st.session_state.answer_generator = AnswerGenerator()

if "comparator" not in st.session_state:
    st.session_state.comparator = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []

if "last_comparison" not in st.session_state:
    st.session_state.last_comparison = None


manager = st.session_state.document_manager


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown('<div class="brand">VERILEX AI</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="brand-subtitle">LEGAL INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Document Library",
            "Ask VERILEX",
            "Compare Documents",
            "Evidence Explorer"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.caption("POWERED BY LOCAL AI")
    st.caption("Qwen 2.5 · Hybrid Retrieval")
    st.caption("Evidence-first analysis")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def build_index():
    with st.spinner("Building hybrid search index..."):
        manager.build_index()
        st.session_state.indexed = True


def render_metrics():
    documents = manager.get_documents()
    total_pages = sum(d["total_pages"] for d in documents)

    col1, col2, col3 = st.columns(3)

    metrics = [
        (col1, "DOCUMENTS", len(documents)),
        (col2, "TOTAL PAGES", total_pages),
        (col3, "INDEX STATUS",
         "Ready" if manager.is_indexed else "Pending")
    ]

    for column, label, value in metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_citations(citations):
    if not citations:
        return

    st.markdown("**Sources**")

    for citation in citations:
        article = citation.get("article")

        page_numbers = citation.get("page_numbers") or [
            citation["page_number"]
        ]

        page_numbers = sorted(set(page_numbers))

        if len(page_numbers) == 1:
            page_label = f"Page {page_numbers[0]}"
        else:
            page_label = (
                f"Pages {', '.join(map(str, page_numbers))}"
            )

        source_label = (
            f"{citation['document_name']} · "
            f"{page_label}"
        )

        if article:
            source_label += f" · {article}"

        st.markdown(f"- 📄 {source_label}")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>Legal intelligence, grounded in evidence.</h1>
        <p>
        Analyse legal documents, uncover relationships,
        and receive answers supported by traceable evidence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_metrics()

    st.markdown("")
    st.markdown(
        '<div class="section-title">Your workspace</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📚 Document Intelligence")
        st.write(
            "Upload legal PDFs, extract their structure, "
            "and create a searchable document collection."
        )

        st.markdown("### ⚖️ Legal Question Answering")
        st.write(
            "Ask questions and receive answers grounded "
            "in retrieved legal passages."
        )

    with col2:
        st.markdown("### 🔍 Cross-Document Analysis")
        st.write(
            "Compare multiple documents to discover "
            "similarities, differences, and potential conflicts."
        )

        st.markdown("### 🧾 Evidence Explorer")
        st.write(
            "Inspect retrieved passages and trace each answer "
            "back to its source document and page."
        )

    st.info(
        "Start by uploading your legal PDFs in the Document Library."
    )


# --------------------------------------------------
# DOCUMENT LIBRARY
# --------------------------------------------------

elif page == "Document Library":

    st.markdown(
        '<div class="section-title">Document Library</div>',
        unsafe_allow_html=True
    )

    st.write("Upload one or multiple legal PDF documents.")

    uploaded_files = st.file_uploader(
        "Choose legal PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Process uploaded documents",
        type="primary",
        disabled=not uploaded_files
    ):

        progress = st.progress(0)
        successful = 0

        for index, uploaded_file in enumerate(uploaded_files):

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                result = manager.add_document(
                    temp_path,
                    document_name=uploaded_file.name
                )
                successful += 1

                st.success(
                    f"Processed {result['document_name']} · "
                    f"{result['total_pages']} pages · "
                    f"{result['total_chunks']} chunks"
                )

            except ValueError as error:
                st.warning(str(error))

            except Exception as error:
                st.error(
                    f"Could not process {uploaded_file.name}: {error}"
                )

            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink()

            progress.progress((index + 1) / len(uploaded_files))

        if successful:
            st.session_state.indexed = False
            st.session_state.last_answer = None
            st.session_state.last_comparison = None
            st.session_state.last_evidence = []

            try:
                build_index()
                st.success("Document collection indexed successfully.")
            except Exception as error:
                st.error(f"Indexing failed: {error}")

    st.markdown("---")
    st.markdown("### Your uploaded documents")

    documents = manager.get_documents()

    if documents:

        for document in documents:

            with st.container(border=True):
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"**📄 {document['document_name']}**")
                    st.caption(
                        f"{document['total_pages']} pages · "
                        f"{document['total_headings']} detected headings"
                    )

                with col2:
                    st.caption("Indexed" if manager.is_indexed else "Pending")

        render_metrics()

    else:
        st.info("Your document library is currently empty.")


# --------------------------------------------------
# ASK VERILEX
# --------------------------------------------------

elif page == "Ask VERILEX":

    st.markdown(
        '<div class="section-title">Ask VERILEX</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Ask a legal question. VERILEX will retrieve relevant passages "
        "and generate an answer using only the available evidence."
    )

    if not manager.get_documents():
        st.warning("Upload documents in the Document Library first.")

    elif not manager.is_indexed:
        st.warning("The document index needs to be built.")
        if st.button("Build search index"):
            try:
                build_index()
                st.rerun()
            except Exception as error:
                st.error(str(error))

    else:

        question = st.text_area(
            "Your legal question",
            placeholder=(
                "Example: What principles apply to the processing "
                "of personal data?"
            ),
            height=110
        )

        top_k = st.slider(
            "Evidence passages to retrieve",
            min_value=3,
            max_value=10,
            value=5
        )

        if st.button("Analyse question", type="primary"):

            if not question.strip():
                st.warning("Please enter a question.")

            else:
                try:
                    with st.spinner(
                        "Retrieving evidence and analysing your question..."
                    ):

                        start_time = time.perf_counter()

                        evidence = manager.search(
                            question,
                            top_k=top_k
                        )

                        answer = st.session_state.answer_generator.generate(
                            question,
                            evidence
                        )

                        elapsed = time.perf_counter() - start_time

                    st.session_state.last_answer = answer
                    st.session_state.last_evidence = evidence
                    st.session_state.last_latency = elapsed

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

        answer = st.session_state.last_answer

        if answer:

            st.markdown("---")
            st.markdown("### VERILEX analysis")

            if answer["abstained"]:
                st.warning(answer["answer"])

            else:
                st.success(answer["answer"])

                render_citations(answer["citations"])

            st.caption(
                f"Analysis time: "
                f"{st.session_state.get('last_latency', 0):.2f} seconds"
            )


# --------------------------------------------------
# COMPARE DOCUMENTS
# --------------------------------------------------

elif page == "Compare Documents":

    st.markdown(
        '<div class="section-title">Compare Documents</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare legal provisions across documents using "
        "evidence retrieved from each source."
    )

    documents = manager.get_documents()

    if len(documents) < 2:
        st.warning(
            "Upload and index at least two documents to use comparison."
        )

    elif not manager.is_indexed:
        st.warning("Build the document index before comparing.")
        if st.button("Build search index"):
            try:
                build_index()
                st.rerun()
            except Exception as error:
                st.error(str(error))

    else:

        document_options = {
            f"{document['document_name']} ({document['document_id'][:8]})":
                document["document_id"]
            for document in documents
        }

        selected_documents = st.multiselect(
            "Documents to compare",
            options=list(document_options.keys()),
            default=list(document_options.keys())
        )

        topic = st.text_input(
            "Comparison topic",
            placeholder=(
                "Example: Data protection and data governance"
            )
        )

        if st.button("Compare documents", type="primary"):

            if len(selected_documents) < 2:
                st.warning("Select at least two documents.")

            elif not topic.strip():
                st.warning("Enter a comparison topic.")

            else:
                try:
                    with st.spinner(
                        "Comparing evidence across documents..."
                    ):

                        comparator = DocumentComparator(manager)

                        result = comparator.compare(
                            topic,
                            document_ids=[
                                document_options[label]
                                for label in selected_documents
                            ]
                        )

                    st.session_state.last_comparison = result

                except Exception as error:
                    st.error(f"Comparison failed: {error}")

        result = st.session_state.last_comparison

        if result:

            st.markdown("---")
            st.markdown("### Comparison results")

            st.write(result["summary"])

            if result["abstained"]:
                st.warning(
                    "There is insufficient cross-document evidence "
                    "to support a comparison."
                )

            else:

                tabs = st.tabs([
                    "Similarities",
                    "Differences",
                    "Potential conflicts"
                ])

                sections = [
                    ("similarities", tabs[0]),
                    ("differences", tabs[1]),
                    ("potential_conflicts", tabs[2])
                ]

                for key, tab in sections:

                    with tab:

                        items = result[key]

                        if not items:
                            st.info("No validated findings in this category.")

                        for item in items:

                            with st.container(border=True):
                                st.markdown(f"**{item['claim']}**")

                                if item.get("reason"):
                                    st.write(item["reason"])

                                render_citations(item["citations"])

                st.markdown("### Overall assessment")
                st.write(result["overall_assessment"])

                st.caption(
                    f"Evidence passages analysed: {result['evidence_count']}"
                )


# --------------------------------------------------
# EVIDENCE EXPLORER
# --------------------------------------------------

elif page == "Evidence Explorer":

    st.markdown(
        '<div class="section-title">Evidence Explorer</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Inspect the actual passages retrieved by VERILEX AI "
        "and understand where its evidence comes from."
    )

    evidence = st.session_state.last_evidence

    if not evidence:
        st.info(
            "Ask a question in the Ask VERILEX workspace first. "
            "Retrieved passages will appear here."
        )

    else:

        st.markdown(
            f"**{len(evidence)} passages retrieved**"
        )

        for index, item in enumerate(evidence, start=1):

            page_numbers = item.get("page_numbers") or [
                item["page_number"]
            ]

            page_numbers = sorted(set(page_numbers))

            page_label = (
                f"Page {page_numbers[0]}"
                if len(page_numbers) == 1
                else f"Pages {', '.join(map(str, page_numbers))}"
            )

            with st.expander(
                f"Evidence {index} · {item['document_name']} · "
                f"{page_label}",
                expanded=index == 1
            ):

                st.caption(
                    f"Document: {item['document_name']}"
                )

                st.caption(page_label)

                if item.get("article"):
                    st.caption(f"Article: {item['article']}")

                st.write(item["text"])


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8290A3; font-size:12px;">
        VERILEX AI · Evidence-grounded legal document analysis
        <br>
        Local AI · Hybrid Retrieval · Transparent Evidence
    </div>
    """,
    unsafe_allow_html=True
)
