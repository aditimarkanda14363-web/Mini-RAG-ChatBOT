"""
Tests for document loader (PDF & TXT).
"""

from pathlib import Path
import pytest
from src.document_loader import clean_text, load_document, load_pdf, load_txt


@pytest.fixture
def sample_pdf_path():
    path = Path(__file__).resolve().parent.parent / "sample_documents" / "artificial_intelligence_overview.pdf"
    assert path.exists(), f"Sample PDF missing at {path}"
    return path


@pytest.fixture
def sample_txt_path():
    path = Path(__file__).resolve().parent.parent / "sample_documents" / "prompt_engineering_guide.txt"
    assert path.exists(), f"Sample TXT missing at {path}"
    return path


def test_clean_text():
    raw = "Hello   world!\r\n\r\n\n\nThis is    a test."
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_load_pdf(sample_pdf_path):
    docs = load_pdf(sample_pdf_path, filename=sample_pdf_path.name)
    assert len(docs) == 3  # 3 pages
    for i, doc in enumerate(docs):
        assert doc.metadata["page"] == i + 1
        assert doc.metadata["source"] == sample_pdf_path.name
        assert doc.metadata["total_pages"] == 3
        assert len(doc.page_content) > 50


def test_load_txt(sample_txt_path):
    docs = load_txt(sample_txt_path, filename=sample_txt_path.name)
    assert len(docs) == 1
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == sample_txt_path.name
    assert "Prompt Engineering" in docs[0].page_content


def test_load_document_dispatcher(sample_pdf_path, sample_txt_path):
    pdf_docs = load_document(sample_pdf_path, filename=sample_pdf_path.name)
    assert len(pdf_docs) == 3

    txt_docs = load_document(sample_txt_path, filename=sample_txt_path.name)
    assert len(txt_docs) == 1

    with pytest.raises(ValueError):
        load_document("dummy.docx", filename="dummy.docx")
