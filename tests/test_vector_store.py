"""
Tests for FAISS vector store.
"""

from pathlib import Path
import numpy as np
import pytest
from src.document_loader import Document
from src.vector_store import FAISSVectorStore


def test_vector_store_add_and_search(tmp_path):
    store = FAISSVectorStore(dimension=4)

    # 3 dummy documents with known normalized vectors
    docs = [
        Document(page_content="Apple fruit", metadata={"source": "fruit.txt", "page": 1}),
        Document(page_content="Banana fruit", metadata={"source": "fruit.txt", "page": 1}),
        Document(page_content="Car vehicle", metadata={"source": "auto.txt", "page": 1})
    ]

    # Create dummy normalized embeddings
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    v2 /= np.linalg.norm(v2)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    embeddings = np.vstack([v1, v2, v3])
    store.add_documents(docs, embeddings)

    assert store.count() == 3

    # Query with vector close to Apple
    query_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = store.similarity_search_by_vector(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0][0].page_content == "Apple fruit"
    assert np.isclose(results[0][1], 1.0, atol=1e-3)
    assert results[1][0].page_content == "Banana fruit"


def test_vector_store_persistence(tmp_path):
    save_dir = tmp_path / "test_faiss"
    store = FAISSVectorStore(dimension=3)

    docs = [
        Document(page_content="Neural Networks", metadata={"source": "ai.pdf", "page": 3})
    ]
    vecs = np.array([[0.577, 0.577, 0.577]], dtype=np.float32)
    store.add_documents(docs, vecs)

    store.save_local(save_dir)

    # Load back
    new_store = FAISSVectorStore(dimension=3)
    new_store.load_local(save_dir)

    assert new_store.count() == 1
    assert new_store.documents[0].page_content == "Neural Networks"
    assert new_store.documents[0].metadata["page"] == 3
