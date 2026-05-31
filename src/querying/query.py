"""RAG query pipeline: retrieve similar chunks and generate LLM answers."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import CHUNKS_PATH, FAISS_INDEX_PATH, TOP_K
from src.indexing.embeddings import generate_embeddings
from src.querying.evaluator import evaluate_answer
from src.querying.input_validation import validate_question
from src.utils.output_manager import save_query_result

SYSTEM_PROMPT = """You are a FAQ assistant.

Answer only using the provided context.

If the answer cannot be found in the provided context, state that the information is not available in the FAQ document.

Do not invent information.

Keep answers concise and factual."""

_SPINNER_FRAMES = ("|", "/", "-", "\\")


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
    evaluation: dict,
) -> dict:
    """Build the final RAG response payload.

    Args:
        question: Original user question.
        answer: LLM-generated answer.
        chunks: Retrieved chunks with scores.
        evaluation: Evaluator result with score and reason.

    Returns:
        Response dictionary with user_question, system_answer, chunks_related,
        and evaluation.
    """
    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": chunks,
        "evaluation": evaluation,
    }


def run_query_pipeline(
    question: str,
    index,
    chunks: list[dict],
) -> dict:
    """Execute retrieval, generation, and evaluation for one question.

    Args:
        question: User question text.
        index: Loaded FAISS index.
        chunks: All chunk records from chunks.json.

    Returns:
        Complete RAG response payload.
    """
    question_embedding = embed_query(question)
    similar_chunks = search_similar_chunks(
        question_embedding, index, chunks, TOP_K
    )
    context = build_context(similar_chunks)
    answer = generate_answer(question, context)
    evaluation = evaluate_answer(question, answer, similar_chunks)
    return build_response(question, answer, similar_chunks, evaluation)


def _run_spinner(stop_event: threading.Event) -> None:
    """Display a console loading spinner until stop_event is set.

    Args:
        stop_event: Threading event that signals spinner shutdown.
    """
    frame_index = 0
    while not stop_event.is_set():
        frame = _SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]
        sys.stdout.write(f"\rSearching... {frame}")
        sys.stdout.flush()
        frame_index += 1
        time.sleep(0.1)


def run_with_spinner(pipeline_fn) -> dict:
    """Run a pipeline function while showing a loading spinner.

    Args:
        pipeline_fn: Callable that returns the pipeline response dict.

    Returns:
        Response returned by pipeline_fn.
    """
    stop_event = threading.Event()
    spinner_thread = threading.Thread(
        target=_run_spinner,
        args=(stop_event,),
        daemon=True,
    )
    spinner_thread.start()
    try:
        return pipeline_fn()
    finally:
        stop_event.set()
        spinner_thread.join()
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()


def main() -> None:
    """Run the interactive RAG query pipeline."""
    print("\n===== Hover HR RAG Query CLI =====\n")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to quit.\n")

    project_root = Path(__file__).resolve().parent.parent.parent
    index_path = project_root / FAISS_INDEX_PATH
    chunks_path = project_root / CHUNKS_PATH

    index = load_faiss_index(str(index_path))
    chunks = load_chunks(str(chunks_path))

    while True:
        try:
            question = input("Question: ").strip()
        except KeyboardInterrupt:
            print()
            print("\nBye 👋\n")
            break

        if question.lower() in ("quit", "exit"):
            print("\nBye 👋\n")
            break

        is_valid, error_message = validate_question(question)
        if not is_valid:
            print(error_message)
            continue

        response = run_with_spinner(
            lambda: run_query_pipeline(question, index, chunks)
        )
        print(json.dumps(response, indent=2, ensure_ascii=False))
        save_query_result(response)


if __name__ == "__main__":
    main()
