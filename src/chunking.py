"""Document loading and word-based sliding-window chunking for RAG indexing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_document(file_path: str) -> str:
    """Read a UTF-8 text file and return its full contents.

    Args:
        file_path: Path to the document to load.

    Returns:
        The complete file contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")
    return path.read_text(encoding="utf-8")


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word-based chunks using a sliding window.

    Args:
        text: Source document text.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        Ordered list of chunk strings. Each index is the logical chunk id.
    """
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


def get_chunk_statistics(chunks: list[str]) -> dict:
    """Compute word-count statistics for a list of chunks.

    Args:
        chunks: List of text chunks.

    Returns:
        Dictionary with total_chunks, min_words, max_words, and avg_words.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "min_words": 0,
            "max_words": 0,
            "avg_words": 0.0,
        }

    word_counts = [len(chunk.split()) for chunk in chunks]
    total = len(word_counts)

    return {
        "total_chunks": total,
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "avg_words": sum(word_counts) / total,
    }


if __name__ == "__main__":
    document_path = Path(__file__).resolve().parent.parent / "data" / "faq_document.txt"
    content = load_document(str(document_path))
    chunks = create_chunks(content)
    stats = get_chunk_statistics(chunks)

    print("Chunking statistics:")
    print(f"  total_chunks: {stats['total_chunks']}")
    print(f"  min_words:    {stats['min_words']}")
    print(f"  max_words:    {stats['max_words']}")
    print(f"  avg_words:    {stats['avg_words']:.2f}")