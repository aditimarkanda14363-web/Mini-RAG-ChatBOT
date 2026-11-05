"""
RAG Pipeline Orchestrator
Coordinates document loading, chunking, embedding generation, FAISS vector indexing,
semantic similarity search, context framing, and local LLM answer generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import DEFAULT_CONFIG, RAGConfig
from src.document_loader import Document, load_document
from src.embedding_manager import EmbeddingManager
from src.llm_client import OllamaClient
from src.prompt_templates import (
    RAG_SYSTEM_PROMPT,
    build_conversational_rag_prompt,
    build_rag_prompt,
)
from src.text_splitter import RecursiveCharacterTextSplitter
from src.vector_store import FAISSVectorStore


@dataclass
class SourceCitation:
    """Represents a cited source snippet with metadata."""
    source: str
    page: int
    score: float
    snippet: str
    chunk_id: str


@dataclass
class QueryContext:
    """Prepared retrieval context ready for generation or refusal."""
    question: str
    is_ready: bool
    is_relevant: bool
    citations: List[SourceCitation]
    system_prompt: str
    prompt: str
    context_blocks: str
    active_model: str
    temperature: float
    immediate_answer: Optional[str] = None
    top_score: float = 0.0


@dataclass
class RAGResult:
    """Encapsulates the complete RAG output, citations, and prompt engineering telemetry."""
    question: str
    answer: str
    sources: List[SourceCitation] = field(default_factory=list)
    system_prompt: str = ""
    prompt_used: str = ""
    context_used: str = ""
    sufficient_info: bool = True
    model_name: str = ""


class RAGPipeline:
    """
    Complete end-to-end local Retrieval-Augmented Generation pipeline.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model_name
        )
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_manager.dimension
        )
        self.llm_client = OllamaClient(
            base_url=self.config.ollama_base_url
        )
        self.processed_files: List[str] = []

    def ingest_document(
        self,
        source: Any,
        filename: str
    ) -> Dict[str, Any]:
        """
        Loads, splits, embeds, and indexes a single document (PDF or TXT).
        
        Args:
            source: Path, BytesIO, or file-like object.
            filename: File name for metadata.
            
        Returns:
            Dict summarizing ingestion metrics (pages, chunks, dimensions).
        """
        # 1. Extract text and page metadata
        raw_docs = load_document(source, filename=filename)
        if not raw_docs:
            return {"filename": filename, "pages": 0, "chunks": 0, "status": "empty"}

        # 2. Chunk text
        chunked_docs = self.text_splitter.split_documents(raw_docs)
        if not chunked_docs:
            return {"filename": filename, "pages": len(raw_docs), "chunks": 0, "status": "no_chunks"}

        # 3. Generate embeddings
        texts = [doc.page_content for doc in chunked_docs]
        embeddings = self.embedding_manager.embed_texts(texts)

        # 4. Add to vector store
        self.vector_store.add_documents(chunked_docs, embeddings)
        if filename not in self.processed_files:
            self.processed_files.append(filename)

        return {
            "filename": filename,
            "pages": len(raw_docs),
            "chunks": len(chunked_docs),
            "status": "success"
        }

    def ingest_documents(
        self,
        file_list: List[Tuple[Any, str]]
    ) -> List[Dict[str, Any]]:
        """Ingests multiple documents into the vector store."""
        results = []
        for source, filename in file_list:
            res = self.ingest_document(source, filename)
            results.append(res)
        return results

    def prepare_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        chat_history: Optional[List[dict]] = None
    ) -> QueryContext:
        """
        Executes semantic search, computes relevance, formats citations, and constructs
        the grounded RAG prompt, returning a QueryContext for either streaming or synchronous execution.
        """
        k = top_k if top_k is not None else self.config.top_k
        active_model = model_name or self.config.default_llm_model
        temp = temperature if temperature is not None else self.config.temperature

        if self.vector_store.count() == 0:
            return QueryContext(
                question=question,
                is_ready=False,
                is_relevant=False,
                citations=[],
                system_prompt=RAG_SYSTEM_PROMPT,
                prompt="N/A (No documents loaded)",
                context_blocks="N/A",
                active_model=active_model,
                temperature=temp,
                immediate_answer=(
                    "No documents have been indexed yet. "
                    "Please upload a PDF or TXT document first so I can retrieve relevant context."
                ),
                top_score=0.0
            )

        # 1. Embed query and search FAISS index
        query_vector = self.embedding_manager.embed_query(question)
        search_results = self.vector_store.similarity_search_by_vector(
            query_vector=query_vector,
            top_k=k,
            score_threshold=None
        )

        top_score = search_results[0][1] if search_results else 0.0
        is_relevant = bool(search_results and top_score >= self.config.similarity_threshold)

        # 2. Build citations
        citations: List[SourceCitation] = []
        for doc, score in search_results:
            citations.append(
                SourceCitation(
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page", 1),
                    score=float(score),
                    snippet=doc.page_content,
                    chunk_id=doc.metadata.get("chunk_id", "")
                )
            )

        # 3. Build prompt
        prompt, context_blocks = build_conversational_rag_prompt(
            question=question,
            retrieved_documents=search_results,
            chat_history=chat_history
        )

        # 4. Immediate refusal if below similarity threshold
        immediate_answer = None
        if not is_relevant:
            immediate_answer = (
                f"I cannot find sufficient information in the provided document(s) to answer this question.\n\n"
                f"The highest similarity score for your query was {top_score:.2f}, which is below our relevance threshold ({self.config.similarity_threshold:.2f}). "
                f"The uploaded document does not appear to discuss '{question.strip()}'."
            )

        return QueryContext(
            question=question,
            is_ready=True,
            is_relevant=is_relevant,
            citations=citations,
            system_prompt=RAG_SYSTEM_PROMPT,
            prompt=prompt,
            context_blocks=context_blocks,
            active_model=active_model,
            temperature=temp,
            immediate_answer=immediate_answer,
            top_score=top_score
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        chat_history: Optional[List[dict]] = None
    ) -> RAGResult:
        """
        Executes semantic search, constructs the grounded prompt, and queries the local LLM synchronously.
        """
        ctx = self.prepare_query(
            question=question,
            top_k=top_k,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history
        )

        if not ctx.is_ready or not ctx.is_relevant:
            return RAGResult(
                question=ctx.question,
                answer=ctx.immediate_answer or "No answer available.",
                sources=ctx.citations,
                system_prompt=ctx.system_prompt,
                prompt_used=ctx.prompt,
                context_used=ctx.context_blocks,
                sufficient_info=False,
                model_name=ctx.active_model
            )

        # Generate answer via Ollama LLM
        try:
            llm_response = self.llm_client.generate(
                prompt=ctx.prompt,
                system_prompt=ctx.system_prompt,
                model=ctx.active_model,
                temperature=ctx.temperature
            )
            raw_answer = llm_response.strip()
            
            # Check if LLM explicitly refused
            sufficient = not (
                "cannot find sufficient information" in raw_answer.lower()
                or "does not contain" in raw_answer.lower()
                or "insufficient information" in raw_answer.lower()
            )

            return RAGResult(
                question=ctx.question,
                answer=raw_answer,
                sources=ctx.citations,
                system_prompt=ctx.system_prompt,
                prompt_used=ctx.prompt,
                context_used=ctx.context_blocks,
                sufficient_info=sufficient,
                model_name=ctx.active_model
            )

        except Exception as e:
            return RAGResult(
                question=ctx.question,
                answer=f"⚠️ LLM Generation Error: {str(e)}",
                sources=ctx.citations,
                system_prompt=ctx.system_prompt,
                prompt_used=ctx.prompt,
                context_used=ctx.context_blocks,
                sufficient_info=False,
                model_name=ctx.active_model
            )

    def reset(self) -> None:
        """Clears the vector store and resets processed files."""
        self.vector_store.clear()
        self.processed_files.clear()
