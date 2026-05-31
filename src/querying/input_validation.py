"""Lightweight input validation for the RAG query CLI."""

from src.config import MAX_QUERY_LENGTH

BLOCK_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "you are now",
    "act as",
    "forget everything",
]

EMPTY_QUESTION_MESSAGE = "Please enter a question."
TOO_LONG_MESSAGE = (
    f"Your question exceeds the maximum length of {MAX_QUERY_LENGTH} characters."
)
INJECTION_MESSAGE = (
    "Your question could not be processed. Please rephrase your HR-related question."
)


def suspicious_input_detected(question: str) -> bool:
    """Detect obvious prompt-injection patterns in user input.

    Args:
        question: User question text.

    Returns:
        True if a blocked pattern is found, False otherwise.
    """
    normalized = question.lower()
    return any(pattern in normalized for pattern in BLOCK_PATTERNS)


def validate_question(question: str) -> tuple[bool, str | None]:
    """Validate user input before retrieval and generation.

    Args:
        question: Raw user question text.

    Returns:
        Tuple of (is_valid, error_message). error_message is None when valid.
    """
    if not question.strip():
        return False, EMPTY_QUESTION_MESSAGE

    if len(question) > MAX_QUERY_LENGTH:
        return False, TOO_LONG_MESSAGE

    if suspicious_input_detected(question):
        return False, INJECTION_MESSAGE

    return True, None
