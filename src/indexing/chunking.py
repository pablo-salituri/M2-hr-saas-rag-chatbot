"""Document loading and word-based sliding-window chunking for RAG indexing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_document(file_path: str) -> str:
    """ 
    Reads a UTF-8 text file and return its full contents.
    Returns ordered list of chunk strings. Each index is the logical chunk id. 
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")
    return path.read_text(encoding="utf-8")


def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step

    return chunks