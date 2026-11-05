"""
Tests for Sentence Transformers embedding generation.
"""

import numpy as np
import pytest
from src.embedding_manager import EmbeddingManager


@pytest.fixture(scope="module")
def embedding_mgr():
    return EmbeddingManager(model_name="all-MiniLM-L6-v2")


def test_embedding_dimension(embedding_mgr):
    assert embedding_mgr.dimension == 384


def test_embed_texts_normalization(embedding_mgr):
    texts = [
        "Retrieval-Augmented Generation enhances language model grounding.",
        "FAISS is an efficient library for dense vector similarity search."
    ]
    embeddings = embedding_mgr.embed_texts(texts)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # Verify L2 normalization: norm of each vector must be approximately 1.0
    for vec in embeddings:
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-3)


def test_embed_query(embedding_mgr):
    query = "What is prompt engineering?"
    vec = embedding_mgr.embed_query(query)
    assert vec.shape == (1, 384)
    assert np.isclose(np.linalg.norm(vec[0]), 1.0, atol=1e-3)


def test_embed_empty_query(embedding_mgr):
    with pytest.raises(ValueError):
        embedding_mgr.embed_query("   ")
