"""Embedding generation via Ollama (nomic-embed-text).

Provides batch embedding for chunks and single embedding for queries.
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# nomic-embed-text produces 768-dimensional embeddings
EMBEDDING_DIM = 768


def embed_texts(
    texts: list[str],
    model: str | None = None,
    ollama_host: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts using Ollama.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name. Defaults to settings.ollama_embed_model.
        ollama_host: Ollama API host. Defaults to settings.ollama_host.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = model or settings.ollama_embed_model
    host = ollama_host or settings.ollama_host

    try:
        response = httpx.post(
            f"{host}/api/embed",
            json={"model": model, "input": texts},
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])

        if len(embeddings) != len(texts):
            logger.warning(
                f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            )

        return embeddings

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


def embed_query(
    query: str,
    model: str | None = None,
    ollama_host: str | None = None,
) -> list[float]:
    """Generate an embedding for a single query string.

    Args:
        query: The search query text.
        model: Embedding model name.
        ollama_host: Ollama API host.

    Returns:
        Embedding vector as a list of floats.
    """
    results = embed_texts([query], model=model, ollama_host=ollama_host)
    if not results:
        raise ValueError("Failed to generate query embedding")
    return results[0]


def embed_chunks_batched(
    texts: list[str],
    batch_size: int = 32,
    model: str | None = None,
    ollama_host: str | None = None,
) -> list[list[float]]:
    """Generate embeddings in batches to avoid overloading Ollama.

    Args:
        texts: All texts to embed.
        batch_size: Number of texts per API call.
        model: Embedding model name.
        ollama_host: Ollama API host.

    Returns:
        List of embedding vectors in the same order as input texts.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(
            f"Embedding batch {i // batch_size + 1}/"
            f"{(len(texts) + batch_size - 1) // batch_size} "
            f"({len(batch)} texts)"
        )
        embeddings = embed_texts(batch, model=model, ollama_host=ollama_host)
        all_embeddings.extend(embeddings)

    return all_embeddings
