"""FAISS index construction and persistence for chunk vectors."""

import json
from pathlib import Path

import faiss
import numpy as np


def build_faiss_index(embeddings: list[list[float]]):
    # Returns a FAISS index containing all normalized vectors.

    if not embeddings:
        raise ValueError("embeddings must not be empty")

    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return index


def save_faiss_index(index, path: str) -> None:
    # Persist a FAISS index to disk.

    faiss.write_index(index, path)


def save_chunks(chunks: list[str], path: str) -> None:
    # Persist chunks as JSON with logical chunk ids.

    payload = [
        {"chunk_id": i, "text": text}
        for i, text in enumerate(chunks)
    ]
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
