"""Pipeline to chunk a document, embed chunks, and persist FAISS + metadata."""

import sys
from pathlib import Path

# Allow imports when run as: python src/build_index.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import create_chunks, load_document
from src.config import CHUNKS_PATH, FAQ_DOCUMENT_PATH, FAISS_INDEX_PATH, STORAGE_DIR
from src.embeddings import generate_embeddings
from src.vector_store import build_faiss_index, save_chunks, save_faiss_index


def main() -> None:
    """Load document, build embeddings and FAISS index, save artifacts."""
    project_root = Path(__file__).resolve().parent.parent
    document_path = project_root / FAQ_DOCUMENT_PATH
    storage_dir = project_root / STORAGE_DIR
    index_path = project_root / FAISS_INDEX_PATH
    chunks_path = project_root / CHUNKS_PATH

    content = load_document(str(document_path))
    print("Document loaded")

    chunks = create_chunks(content)
    print(f"Chunks generated: {len(chunks)}")

    embeddings = generate_embeddings(chunks)
    print(f"Embeddings generated: {len(embeddings)}")

    index = build_faiss_index(embeddings)

    storage_dir.mkdir(parents=True, exist_ok=True)
    save_faiss_index(index, str(index_path))
    print("FAISS index saved")

    save_chunks(chunks, str(chunks_path))
    print("Chunks saved")


if __name__ == "__main__":
    main()