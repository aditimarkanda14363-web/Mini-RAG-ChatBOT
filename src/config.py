"""
Configuration Module for Mini RAG Chatbot
Defines default settings, model paths, chunking parameters, and vector store configurations.
"""

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "faiss_index"


@dataclass
class RAGConfig:
    # Model settings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    default_llm_model: str = "llama3.2:3b"
    
    # Text splitting settings
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Retrieval settings
    top_k: int = 3
    similarity_threshold: float = 0.35  # Cosine similarity cutoff for relevance
    
    # Generation settings
    temperature: float = 0.1  # Low temperature for factual, grounded answers
    max_tokens: int = 512
    
    # Fallback response for ungrounded queries
    insufficient_info_message: str = (
        "I cannot find sufficient information in the provided document(s) to answer this question. "
        "The uploaded text does not contain relevant details regarding your query."
    )


# Global default configuration instance
DEFAULT_CONFIG = RAGConfig()
