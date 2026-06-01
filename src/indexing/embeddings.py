"""OpenAI embedding generation for RAG indexing."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    # Returns list of embedding vectors in the same order as texts.
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("EMBEDDING_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not model:
        raise ValueError("EMBEDDING_MODEL environment variable is not set")

    client = OpenAI(api_key=api_key)

    if not texts:
        return []

    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]
