"""
Vector Store Module
Manages the FAISS index (faiss-cpu) for high-performance semantic similarity search,
vector storage, persistence to disk, and retrieval with cosine similarity scoring.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
import faiss
import numpy as np
from src.document_loader import Document


class FAISSVectorStore:
    """
    In-memory and persistent FAISS vector store using IndexFlatIP (Inner Product).
    When embeddings are L2-normalized, Inner Product corresponds exactly to Cosine Similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Adds text documents and their corresponding vector embeddings into the FAISS index.
        
        Args:
            documents: List of Document instances.
            embeddings: 2D numpy array of shape (len(documents), dimension) with float32 type.
        """
        if len(documents) == 0:
            return

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding matrix shape {embeddings.shape} does not match expected dimension (*, {self.dimension})"
            )

        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) does not match embedding rows ({embeddings.shape[0]})"
            )

        # FAISS expects float32
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_f32)
        self.documents.extend(documents)

    def similarity_search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves the top-k most similar documents based on cosine similarity score.
        
        Args:
            query_vector: 2D array of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.
            score_threshold: Minimum cosine similarity score to qualify (e.g. 0.35).
            
        Returns:
            List of tuples: (Document, similarity_score) sorted from highest to lowest score.
        """
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        k = min(top_k, self.index.ntotal)
        
        distances, indices = self.index.search(query_vector, k)

        results: List[Tuple[Document, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            score = float(dist)
            if score_threshold is not None and score < score_threshold:
                continue
            doc = self.documents[idx]
            results.append((doc, score))

        return results

    def save_local(self, folder_path: Path) -> None:
        """Serializes the FAISS index and documents metadata to a local directory."""
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        index_file = folder_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        docs_file = folder_path / "documents.json"
        docs_payload = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in self.documents
        ]
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(docs_payload, f, ensure_ascii=False, indent=2)

    def load_local(self, folder_path: Path) -> None:
        """Loads a saved FAISS index and associated documents from disk."""
        folder_path = Path(folder_path)
        index_file = folder_path / "index.faiss"
        docs_file = folder_path / "documents.json"

        if not index_file.exists() or not docs_file.exists():
            raise FileNotFoundError(f"Vector store files missing in {folder_path}")

        self.index = faiss.read_index(str(index_file))
        self.dimension = self.index.d

        with open(docs_file, "r", encoding="utf-8") as f:
            docs_payload = json.load(f)

        self.documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in docs_payload
        ]

    def clear(self) -> None:
        """Resets the vector index and clears all stored documents."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []

    def count(self) -> int:
        """Returns the total number of chunks currently indexed."""
        return self.index.ntotal
