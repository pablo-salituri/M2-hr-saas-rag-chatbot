"""RAG query pipeline: retrieve similar chunks and generate LLM answers."""

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNKS_PATH, FAISS_INDEX_PATH, TOP_K
from src.embeddings import generate_embeddings
from src.evaluator import evaluate_answer
from src.output_manager import save_query_result

SYSTEM_PROMPT = """You are an internal HR support assistant.

Answer only using the provided context.

If the answer cannot be found in the context, say:

"The information is not available in the documentation."

Keep answers concise and factual."""


def load_chunks(path: str) -> list[dict]:
    """Load chunk records from a JSON file.

    Args:
        path: Path to chunks.json.

    Returns:
        List of chunk dictionaries with chunk_id and text.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_faiss_index(path: str):
    """Load a persisted FAISS index from disk.

    Args:
        path: Path to faiss.index.

    Returns:
        The loaded FAISS index.
    """
    return faiss.read_index(path)


def embed_query(question: str) -> list[float]:
    """Generate an embedding vector for a user question.

    Args:
        question: User question text.

    Returns:
        Embedding vector for the question.
    """
    return generate_embeddings([question])[0]


def search_similar_chunks(
    question_embedding: list[float],
    index,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """Search FAISS for the most similar chunks to a question embedding.

    Args:
        question_embedding: Query embedding vector.
        index: Loaded FAISS index.
        chunks: All chunk records from chunks.json.
        top_k: Number of results to retrieve.

    Returns:
        Top matching chunks with chunk_id, text, and FAISS similarity score.
    """
    query_vector = np.array([question_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score),
            }
        )

    return results


def build_context(chunks: list[dict]) -> str:
    """Concatenate retrieved chunks into a single context string.

    Args:
        chunks: Retrieved chunks ordered by relevance.

    Returns:
        Formatted context for the LLM prompt.
    """
    parts = [f"Chunk {i}:\n{chunk['text']}" for i, chunk in enumerate(chunks, start=1)]
    return "\n\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using OpenAI chat completions and retrieved context.

    Args:
        question: User question.
        context: Retrieved chunk context.

    Returns:
        LLM-generated answer string.

    Raises:
        ValueError: If required environment variables are missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CHAT_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not model:
        raise ValueError("CHAT_MODEL environment variable is not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def build_response(
    question: str,
    answer: str,
    chunks: list[dict],
) -> dict:
    """Build the final RAG response payload.

    Args:
        question: Original user question.
        answer: LLM-generated answer.
        chunks: Retrieved chunks with scores.

    Returns:
        Response dictionary with user_question, system_answer, and chunks_related.
    """
    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": chunks,
    }


def main() -> None:
    """Run the interactive RAG query pipeline."""
    question = input("Question: ")

    project_root = Path(__file__).resolve().parent.parent
    index_path = project_root / FAISS_INDEX_PATH
    chunks_path = project_root / CHUNKS_PATH

    index = load_faiss_index(str(index_path))
    chunks = load_chunks(str(chunks_path))
    question_embedding = embed_query(question)
    similar_chunks = search_similar_chunks(
        question_embedding, index, chunks, TOP_K
    )
    context = build_context(similar_chunks)
    answer = generate_answer(question, context)
    response = build_response(question, answer, similar_chunks)
    evaluation = evaluate_answer(
        question, answer, similar_chunks
    )
    save_query_result({**response, "evaluation": evaluation})

    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
