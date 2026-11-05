"""
Embedding Manager Module
Encapsulates Sentence Transformers (all-MiniLM-L6-v2) to generate normalized
numerical vector representations for text chunks and search queries.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Manages loading the embedding model and generating L2-normalized vector embeddings.
    Using normalized vectors ensures that Inner Product (IP) similarity equals Cosine Similarity.
    """

    _instance = None
    _model = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Loads the SentenceTransformer model if not already loaded in memory."""
        if EmbeddingManager._model is None:
            # sentence-transformers will download or load from local torch cache
            EmbeddingManager._model = SentenceTransformer(self.model_name)
        self.model = EmbeddingManager._model
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates normalized float32 embeddings for a list of strings.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.
            
        Returns:
            np.ndarray of shape (len(texts), dimension) with L2-normalized vectors.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates normalized float32 embedding for a single search query.
        
        Args:
            query: User's question or search query.
            
        Returns:
            np.ndarray of shape (1, dimension) with L2-normalized vector.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
            
        embedding = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)
