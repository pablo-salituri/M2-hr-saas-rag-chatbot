"""Rule-based evaluator for RAG answer quality."""

import re

NOT_AVAILABLE_PHRASE = "The information is not available in the documentation."

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "how",
        "when",
        "where",
        "who",
        "which",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "many",
        "much",
        "long",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "and",
        "or",
        "at",
        "by",
        "from",
        "it",
        "that",
        "this",
        "their",
        "they",
        "we",
        "you",
        "your",
        "company",
    }
)


def _tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase tokens from text.

    Args:
        text: Input text.

    Returns:
        Set of tokens with stop words removed.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in _STOP_WORDS and len(word) > 1}


def _average_retrieval_score(chunks: list) -> float:
    """Compute the average FAISS score across retrieved chunks.

    Args:
        chunks: Retrieved chunk records.

    Returns:
        Average retrieval score, or 0.0 when scores are unavailable.
    """
    scores = [
        float(chunk["score"])
        for chunk in chunks
        if isinstance(chunk, dict) and "score" in chunk
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _lexical_overlap(source_tokens: set[str], target_text: str) -> float:
    """Measure how much of source_tokens appears in target_text.

    Args:
        source_tokens: Tokens to match against the target.
        target_text: Text to search for overlaps.

    Returns:
        Overlap ratio between 0.0 and 1.0.
    """
    if not source_tokens:
        return 0.0
    target_tokens = _tokenize(target_text)
    if not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(source_tokens)


def _score_relevance(question_tokens: set[str], chunks: list) -> int:
    """Score whether retrieved chunks appear related to the question.

    Args:
        question_tokens: Tokenized question content.
        chunks: Retrieved chunk records.

    Returns:
        Relevance score between 0 and 5.
    """
    if not chunks:
        return 0

    chunk_overlaps = [
        _lexical_overlap(question_tokens, chunk.get("text", ""))
        for chunk in chunks
        if isinstance(chunk, dict)
    ]
    avg_overlap = sum(chunk_overlaps) / len(chunk_overlaps) if chunk_overlaps else 0.0
    avg_retrieval = _average_retrieval_score(chunks)

    score = 0
    if len(chunks) >= 1:
        score += 1
    if len(chunks) >= 2:
        score += 1
    if avg_overlap >= 0.15:
        score += 1
    if avg_overlap >= 0.30:
        score += 1
    if avg_retrieval >= 0.35:
        score += 1

    return min(score, 5)


def _score_completeness(
    question_tokens: set[str],
    answer: str,
    chunks: list,
) -> int:
    """Score whether the answer appears sufficient and grounded in chunks.

    Args:
        question_tokens: Tokenized question content.
        answer: Generated system answer.
        chunks: Retrieved chunk records.

    Returns:
        Completeness score between 0 and 5.
    """
    score = 0
    answer_tokens = _tokenize(answer)
    chunk_text = " ".join(
        chunk.get("text", "") for chunk in chunks if isinstance(chunk, dict)
    )

    if len(answer) >= 20:
        score += 1
    if _lexical_overlap(question_tokens, answer) >= 0.20:
        score += 1
    if _lexical_overlap(answer_tokens, chunk_text) >= 0.20:
        score += 1
    if _lexical_overlap(answer_tokens, chunk_text) >= 0.35:
        score += 1
    if len(answer_tokens) >= 5:
        score += 1

    return min(score, 5)


def evaluate_answer(
    user_question: str,
    system_answer: str,
    chunks_related: list,
) -> dict:
    """Evaluate a RAG response using simple deterministic heuristics.

    Args:
        user_question: Original user question.
        system_answer: Generated answer from the RAG pipeline.
        chunks_related: Retrieved chunks related to the answer.

    Returns:
        Dictionary with integer score (0-10) and textual reason.
    """
    if not user_question or not user_question.strip():
        return {
            "score": 0,
            "reason": "Missing user question; evaluation cannot be performed.",
        }

    if not system_answer or not system_answer.strip():
        return {
            "score": 0,
            "reason": "Missing system answer; evaluation cannot be performed.",
        }

    question = user_question.strip()
    answer = system_answer.strip()
    chunks = chunks_related if chunks_related is not None else []

    if not chunks:
        return {
            "score": 1,
            "reason": "No chunks were retrieved, so the answer lacks supporting context.",
        }

    question_tokens = _tokenize(question)

    if NOT_AVAILABLE_PHRASE in answer:
        avg_retrieval = _average_retrieval_score(chunks)
        if avg_retrieval < 0.30:
            return {
                "score": 4,
                "reason": (
                    "The answer correctly reports unavailable information, "
                    "and retrieved chunks appear weakly related to the question."
                ),
            }
        return {
            "score": 3,
            "reason": (
                "The answer reports unavailable information even though "
                "some retrieved chunks may contain partial context."
            ),
        }

    relevance = _score_relevance(question_tokens, chunks)
    completeness = _score_completeness(question_tokens, answer, chunks)
    score = min(relevance + completeness, 10)

    if score >= 8:
        reason = (
            "The answer is supported by multiple relevant chunks and "
            "appears to address the user's question."
        )
    elif score >= 6:
        reason = (
            "The answer appears reasonably complete and is partially "
            "supported by retrieved chunks."
        )
    elif score >= 4:
        reason = (
            "The answer shows limited support from retrieved chunks or "
            "may not fully address the question."
        )
    else:
        reason = (
            "The answer has weak relevance or completeness relative to "
            "the question and retrieved chunks."
        )

    return {"score": score, "reason": reason}


if __name__ == "__main__":
    result = evaluate_answer(
        user_question="How many failed login attempts are allowed?",
        system_answer=(
            "Five. Reaching five unsuccessful attempts triggers a "
            "30-minute cryptographic lock on the account identifier."
        ),
        chunks_related=[
            {
                "chunk_id": 6,
                "text": (
                    "Reaching five unsuccessful attempts triggers a hard "
                    "cryptographic lock on the account identifier for thirty "
                    "minutes."
                ),
                "score": 0.505,
            },
            {
                "chunk_id": 4,
                "text": (
                    "Password update routines can be self-administered by "
                    "end-users or triggered systematically due to corporate "
                    "expiration lifecycles."
                ),
                "score": 0.383,
            },
            {
                "chunk_id": 3,
                "text": (
                    "Data integrity within human resources systems demands "
                    "rigorous credential protection mechanisms."
                ),
                "score": 0.351,
            },
        ],
    )
    print(result)
