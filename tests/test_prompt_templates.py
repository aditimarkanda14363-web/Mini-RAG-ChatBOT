"""
Tests for Prompt Engineering module and citation framing.
"""

from src.document_loader import Document
from src.prompt_templates import (
    RAG_SYSTEM_PROMPT,
    build_conversational_rag_prompt,
    build_rag_prompt,
    format_context_passage,
)


def test_system_prompt_rules():
    assert "GROUNDED IN CONTEXT ONLY" in RAG_SYSTEM_PROMPT
    assert "HONEST REFUSAL" in RAG_SYSTEM_PROMPT
    assert "ACCURATE CITATIONS" in RAG_SYSTEM_PROMPT


def test_format_context_passage():
    doc = Document(
        page_content="Vaswani et al. introduced the Transformer.",
        metadata={"source": "transformer.pdf", "page": 3, "chunk_id": "c1"}
    )
    formatted = format_context_passage(doc, 1, score=0.88)
    assert "[PASSAGE 1]" in formatted
    assert "Source: transformer.pdf" in formatted
    assert "Page: 3" in formatted
    assert "0.880" in formatted


def test_build_rag_prompt():
    doc = Document(
        page_content="Arthur Samuel coined the term machine learning in 1959.",
        metadata={"source": "ml_history.pdf", "page": 2}
    )
    prompt, context = build_rag_prompt("Who coined machine learning?", [(doc, 0.92)])
    assert "### RETRIEVED CONTEXT PASSAGES ###" in prompt
    assert "Arthur Samuel coined the term machine learning" in prompt
    assert "Who coined machine learning?" in prompt
    assert "### ANSWER ###" in prompt


def test_conversational_rag_prompt():
    doc = Document(page_content="Some test content", metadata={"source": "doc.txt", "page": 1})
    history = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"}
    ]
    prompt, _ = build_conversational_rag_prompt("My follow-up question", [(doc, 0.85)], history)
    assert "### RECENT CONVERSATION HISTORY ###" in prompt
    assert "User: Hello!" in prompt
    assert "Assistant: Hi there! How can I help you?" in prompt
