"""
Document Loader Module
Handles loading and text extraction from PDF (using PyMuPDF) and TXT files,
preserving source file names and page number metadata.
"""

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Dict, List, Union
try:
    import pymupdf as fitz
except ImportError:
    import fitz


@dataclass
class Document:
    """Represents a chunk or page of text with associated metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        snippet = self.page_content[:60].replace("\n", " ") + "..." if len(self.page_content) > 60 else self.page_content
        return f"Document(page={self.metadata.get('page', '?')}, source='{self.metadata.get('source', 'unknown')}', content='{snippet}')"


def clean_text(text: str) -> str:
    """Normalize extracted text by standardizing whitespace while preserving paragraph breaks."""
    if not text:
        return ""
    # Normalize newline characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines into 2
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(source: Union[str, Path, io.BytesIO, bytes], filename: str = "document.pdf") -> List[Document]:
    """
    Extracts text from a PDF file page by page using PyMuPDF.
    
    Args:
        source: File path (str/Path) or in-memory bytes/BytesIO.
        filename: Human-readable name of the document.
        
    Returns:
        List of Document instances, one per non-empty page, with page number metadata.
    """
    documents: List[Document] = []
    
    try:
        if isinstance(source, (str, Path)):
            doc = fitz.open(str(source))
        elif isinstance(source, io.BytesIO):
            doc = fitz.open(stream=source.getvalue(), filetype="pdf")
        elif isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            # Handle Streamlit UploadedFile or file-like objects
            if hasattr(source, "read"):
                content = source.read()
                doc = fitz.open(stream=content, filetype="pdf")
            else:
                raise ValueError(f"Unsupported source type for PDF: {type(source)}")

        total_pages = len(doc)
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            
            if cleaned:
                documents.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": page_index + 1,  # 1-indexed for user readability
                            "total_pages": total_pages,
                            "type": "pdf"
                        }
                    )
                )
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF '{filename}': {str(e)}") from e

    return documents


def load_txt(source: Union[str, Path, io.BytesIO, bytes], filename: str = "document.txt") -> List[Document]:
    """
    Extracts text from a TXT file.
    
    Args:
        source: File path (str/Path) or in-memory bytes/BytesIO.
        filename: Human-readable name of the document.
        
    Returns:
        List containing a single Document instance with page=1.
    """
    raw_content: str = ""
    
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            try:
                raw_content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw_content = path.read_text(encoding="latin-1", errors="replace")
        elif isinstance(source, (io.BytesIO, bytes)):
            byte_data = source.getvalue() if isinstance(source, io.BytesIO) else source
            try:
                raw_content = byte_data.decode("utf-8")
            except UnicodeDecodeError:
                raw_content = byte_data.decode("latin-1", errors="replace")
        elif hasattr(source, "read"):
            data = source.read()
            if isinstance(data, bytes):
                try:
                    raw_content = data.decode("utf-8")
                except UnicodeDecodeError:
                    raw_content = data.decode("latin-1", errors="replace")
            else:
                raw_content = str(data)
        else:
            raise ValueError(f"Unsupported source type for TXT: {type(source)}")

        cleaned = clean_text(raw_content)
        if not cleaned:
            return []
            
        return [
            Document(
                page_content=cleaned,
                metadata={
                    "source": filename,
                    "page": 1,
                    "total_pages": 1,
                    "type": "txt"
                }
            )
        ]
    except Exception as e:
        raise RuntimeError(f"Error reading TXT file '{filename}': {str(e)}") from e


def load_document(source: Any, filename: str) -> List[Document]:
    """
    Detects file type based on extension and delegates to appropriate loader.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return load_pdf(source, filename=filename)
    elif lower_name.endswith(".txt"):
        return load_txt(source, filename=filename)
    else:
        raise ValueError(f"Unsupported file format for '{filename}'. Only .pdf and .txt are supported.")
