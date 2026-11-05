"""
Prompt Engineering Module for RAG Chatbot
Course: Prompt Engineering

This module implements industry-standard prompt engineering techniques:
1. Role & Persona Definition (System Instructions)
2. Context Demarcation & Boundary Enforcement
3. Strict Grounding Directives (Preventing Hallucinations)
4. Explicit Refusal Criteria (Honest Epistemic Refusal)
5. Structured In-Text Citation Protocol
"""

from typing import List, Tuple
from src.document_loader import Document


# System prompt setting the assistant's persona, rules, and grounding constraints
RAG_SYSTEM_PROMPT = """You are a precise, scholarly, and reliable Retrieval-Augmented Generation (RAG) assistant.
Your sole mission is to answer the user's questions truthfully and accurately, strictly based on the provided context passages retrieved from their uploaded documents.

CRITICAL OPERATING RULES:
1. GROUNDED IN CONTEXT ONLY: Rely exclusively on the facts stated directly in the provided context below. Never speculate, assume, or pull information from outside general knowledge.
2. HONEST REFUSAL: If the provided context does not contain sufficient information to answer the question completely and factually, you MUST refuse to answer. State politely:
   "I cannot find sufficient information in the provided document(s) to answer this question. The uploaded text does not discuss [specific topic requested]."
3. ACCURATE CITATIONS: Whenever you state a fact from the text, attribute it to its source using bracketed references, specifying the document name and page number, e.g.:
   "[Source: filename.pdf, Page 3]".
4. NO FABRICATION: Do not invent names, dates, figures, citations, or extrapolate beyond the explicit excerpts.
5. CONCISE & STRUCTURED: Provide clear, well-structured answers using bullet points or short paragraphs when appropriate.
"""


def format_context_passage(doc: Document, index: int, score: float = None) -> str:
    """
    Formats an individual retrieved chunk with clear delimiters and metadata tags.
    """
    source_name = doc.metadata.get("source", "Unknown Document")
    page_num = doc.metadata.get("page", 1)
    chunk_id = doc.metadata.get("chunk_id", f"chunk_{index}")
    
    score_str = f" | Similarity: {score:.3f}" if score is not None else ""
    header = f"--- [PASSAGE {index}] Source: {source_name} | Page: {page_num}{score_str} ---"
    footer = "---"
    return f"{header}\n{doc.page_content.strip()}\n{footer}"


def build_rag_prompt(
    question: str,
    retrieved_documents: List[Tuple[Document, float]],
    chat_history: List[dict] = None
) -> Tuple[str, str]:
    """
    Constructs the grounded RAG user prompt combining context excerpts and the user question.
    
    Returns:
        Tuple of (formatted_full_prompt, formatted_context_only)
    """
    if not retrieved_documents:
        context_blocks = "No relevant passages were found in the uploaded documents."
    else:
        context_blocks = "\n\n".join(
            format_context_passage(doc, idx + 1, score)
            for idx, (doc, score) in enumerate(retrieved_documents)
        )

    prompt = f"""### RETRIEVED CONTEXT PASSAGES ###
{context_blocks}

### INSTRUCTIONS ###
Based ONLY on the context passages above, answer the question below.
- If the answer cannot be determined strictly from the context, state:
  "I cannot find sufficient information in the provided document(s) to answer this question."
- Always include citations with source document name and page number in your response.

### USER QUESTION ###
{question.strip()}

### ANSWER ###"""

    return prompt, context_blocks


def build_conversational_rag_prompt(
    question: str,
    retrieved_documents: List[Tuple[Document, float]],
    chat_history: List[dict] = None,
    max_history_turns: int = 3
) -> Tuple[str, str]:
    """
    Constructs a multi-turn aware prompt with conversation context and retrieved document passages.
    """
    history_str = ""
    if chat_history:
        recent_history = chat_history[- (max_history_turns * 2):]
        turns = []
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            turns.append(f"{role}: {msg.get('content', '')}")
        if turns:
            history_str = "### RECENT CONVERSATION HISTORY ###\n" + "\n".join(turns) + "\n\n"

    prompt, context_blocks = build_rag_prompt(question, retrieved_documents)
    full_prompt = history_str + prompt
    return full_prompt, context_blocks
