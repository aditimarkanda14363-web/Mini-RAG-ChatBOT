"""
Tests for text chunking and splitting.
"""

import pytest
from src.document_loader import Document
from src.text_splitter import RecursiveCharacterTextSplitter


def test_splitter_initialization_validation():
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=150)


def test_split_simple_text():
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    text = "The quick brown fox jumps over the lazy dog. " * 5
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 60  # Allow slight tolerance if words are unbroken


def test_split_documents_metadata_propagation():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    original_doc = Document(
        page_content="Sentence one is here. " * 15,
        metadata={"source": "test.pdf", "page": 2, "type": "pdf"}
    )
    chunked = splitter.split_documents([original_doc])
    assert len(chunked) > 1
    for idx, c in enumerate(chunked):
        assert c.metadata["source"] == "test.pdf"
        assert c.metadata["page"] == 2
        assert "chunk_id" in c.metadata
        assert c.metadata["page_chunk_index"] == idx + 1
        assert c.metadata["char_count"] == len(c.page_content)


def test_split_empty_text():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    assert splitter.split_text("") == []
    assert splitter.split_documents([]) == []
