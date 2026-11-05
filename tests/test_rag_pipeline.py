"""
End-to-end integration tests for RAG pipeline.
"""

from pathlib import Path
import pytest
from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline


@pytest.fixture(scope="module")
def initialized_pipeline():
    config = RAGConfig(
        chunk_size=400,
        chunk_overlap=80,
        similarity_threshold=0.30,
        temperature=0.1
    )
    pipeline = RAGPipeline(config=config)

    # Ingest sample documents
    base_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    pdf_path = base_dir / "artificial_intelligence_overview.pdf"
    txt_path = base_dir / "prompt_engineering_guide.txt"

    assert pdf_path.exists()
    assert txt_path.exists()

    pipeline.ingest_document(pdf_path, filename="artificial_intelligence_overview.pdf")
    pipeline.ingest_document(txt_path, filename="prompt_engineering_guide.txt")

    return pipeline


def test_ingestion_counts(initialized_pipeline):
    assert initialized_pipeline.vector_store.count() > 0
    assert "artificial_intelligence_overview.pdf" in initialized_pipeline.processed_files
    assert "prompt_engineering_guide.txt" in initialized_pipeline.processed_files


def test_factual_query_retrieval(initialized_pipeline):
    # Query regarding page 1 of the PDF
    question = "Who coined the term Artificial Intelligence and at what event in 1956?"
    result = initialized_pipeline.query(question=question)

    assert len(result.sources) > 0
    top_source = result.sources[0]
    assert top_source.source == "artificial_intelligence_overview.pdf"
    assert top_source.page == 1
    assert "Dartmouth" in top_source.snippet

    # Verify answer generation from local Ollama
    assert len(result.answer) > 20
    assert "Dartmouth" in result.answer or "McCarthy" in result.answer
    assert result.sufficient_info is True


def test_page_specific_citation_retrieval(initialized_pipeline):
    # Query regarding page 3 of the PDF (Transformers)
    question = "What is the self-attention formula introduced in the Attention Is All You Need paper?"
    result = initialized_pipeline.query(question=question)

    assert len(result.sources) > 0
    # Page 3 contains Vaswani and Attention Is All You Need
    pages = [s.page for s in result.sources]
    assert 3 in pages


def test_refusal_for_unrelated_query(initialized_pipeline):
    # Query completely unrelated to AI or Prompt Engineering
    question = "How do you bake a triple chocolate fudge cake at 350 degrees?"
    result = initialized_pipeline.query(question=question)

    # Must refuse because similarity score is below threshold or context is insufficient
    assert result.sufficient_info is False
    assert (
        "cannot find sufficient information" in result.answer.lower()
        or "does not contain" in result.answer.lower()
        or "not discuss" in result.answer.lower()
    )
