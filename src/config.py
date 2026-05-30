"""Centralized configuration constants for the RAG pipeline."""

CHUNK_SIZE = 75
CHUNK_OVERLAP = 15

TOP_K = 3

DATA_DIR = "data"
STORAGE_DIR = "storage"

FAQ_DOCUMENT_PATH = "data/faq_document.txt"

FAISS_INDEX_PATH = "storage/faiss.index"
CHUNKS_PATH = "storage/chunks.json"
