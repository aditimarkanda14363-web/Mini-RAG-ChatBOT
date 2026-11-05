"""
Text Splitter Module
Implements recursive semantic chunking with configurable chunk sizes, overlap,
and metadata preservation (source file, page numbers, chunk index).
"""

from typing import List
from src.document_loader import Document


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using a hierarchy of separators (paragraphs, sentences, words, characters)
    to keep semantically related text together while respecting chunk_size and chunk_overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively along the list of separators until chunks fit chunk_size."""
        final_chunks: List[str] = []
        
        # Choose the first separator that appears in text
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        # Merge pieces back up to chunk_size with chunk_overlap
        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    # Can't split further, take as is
                    final_chunks.append(s)
                else:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Combine smaller splits into chunks of up to chunk_size, respecting chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for piece in splits:
            piece_len = len(piece) + (len(separator) if current_doc else 0)
            if total_len + piece_len > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    
                    # Keep overlap pieces from the end of current_doc
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    for prev_piece in reversed(current_doc):
                        p_len = len(prev_piece) + (len(separator) if overlap_doc else 0)
                        if overlap_len + p_len <= self.chunk_overlap:
                            overlap_doc.insert(0, prev_piece)
                            overlap_len += p_len
                        else:
                            break
                    current_doc = overlap_doc
                    total_len = overlap_len
                
            current_doc.append(piece)
            total_len += len(piece) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def split_text(self, text: str) -> List[str]:
        """Splits raw text string into chunks."""
        if not text or not text.strip():
            return []
        return self._split_text(text.strip(), self.separators)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Documents into chunked Documents while preserving and augmenting metadata.
        """
        chunked_docs: List[Document] = []
        global_chunk_idx = 0

        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for page_chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                meta = dict(doc.metadata)
                meta.update({
                    "chunk_id": f"{meta.get('source', 'doc')}_p{meta.get('page', 1)}_c{page_chunk_idx + 1}",
                    "chunk_index": global_chunk_idx,
                    "page_chunk_index": page_chunk_idx + 1,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                })
                chunked_docs.append(Document(page_content=chunk, metadata=meta))
                global_chunk_idx += 1

        return chunked_docs
