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
    # Returns the list of chunk dictionaries with chunk_id and text.

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_faiss_index(path: str):
    # Loads and returns the persisted FAISS index from disk.
    
    return faiss.read_index(path)


def embed_query(question: str) -> list[float]:
    """
    Generates an embedding vector for a user question.
    Returns the embedding vector for the question.
    """
    return generate_embeddings([question])[0]


def search_similar_chunks(
    question_embedding: list[float],
    index,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Searches the FAISS index for the most similar chunks to a question embedding.
    Returns the top matching chunks with chunk_id, text, and FAISS similarity score.
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
    """
    Concatenates the retrieved chunks into a single context string.
    Returns the formatted context for the LLM prompt.
    """
    parts = [f"Chunk {i}:\n{chunk['text']}" for i, chunk in enumerate(chunks, start=1)]
    return "\n\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    """
    Generates an answer using OpenAI chat completions and retrieved context.
    Returns the LLM-generated answer string.
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
    """
    Builds the final RAG response payload.
    Returns a response dictionary with user_question, system_answer, chunks_related and evaluation.
    """
    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": chunks,
        "evaluation": evaluation,
    }


def run_query_pipeline(question: str, index, chunks: list[dict]) -> dict:
    """
    Executes the retrieval, generation, and evaluation for one question.
    Returns the complete RAG response payload.
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
    frame_index = 0
    while not stop_event.is_set():
        frame = _SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]
        sys.stdout.write(f"\rSearching... {frame}")
        sys.stdout.flush()
        frame_index += 1
        time.sleep(0.1)


def run_with_spinner(pipeline_fn) -> dict:
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
    """Runs the query pipeline."""
    print("\n===== HR RAG Query CLI =====\n")
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