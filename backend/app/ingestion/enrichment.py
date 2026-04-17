"""Chunk enrichment pipeline.

Enriches each chunk with:
- TF-IDF keywords (scikit-learn)
- Auto-generated metadata (already set during chunking)
- Chunk summary via Llama 3 (optional, requires Ollama)
- Data lineage tracking
"""

import logging
from dataclasses import dataclass

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.chunks import Chunk, EnrichedChunk

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentConfig:
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    top_k_keywords: int = 10
    generate_summaries: bool = True
    summary_max_tokens: int = 150
    batch_size: int = 20
    summary_timeout: float = 300.0  # seconds per summary request
    summary_connect_timeout: float = 10.0  # seconds for initial connection


def extract_keywords(chunks: list[Chunk], top_k: int = 10) -> dict[str, list[tuple[str, float]]]:
    """Extract TF-IDF keywords for each chunk.

    Args:
        chunks: List of chunks to process.
        top_k: Number of top keywords per chunk.

    Returns:
        Mapping of chunk ID → list of (keyword, score) tuples.
    """
    if not chunks:
        return {}

    texts = [c.text for c in chunks]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        logger.warning("TF-IDF vectorization failed (possibly empty texts)")
        return {c.id: [] for c in chunks}

    feature_names = vectorizer.get_feature_names_out()
    keywords_map: dict[str, list[tuple[str, float]]] = {}

    for i, chunk in enumerate(chunks):
        row = tfidf_matrix[i].toarray().flatten()
        # Get top-k indices by score
        top_indices = row.argsort()[-top_k:][::-1]
        keywords = [
            (feature_names[idx], float(row[idx]))
            for idx in top_indices
            if row[idx] > 0
        ]
        keywords_map[chunk.id] = keywords

    return keywords_map


def _warm_up_model(config: EnrichmentConfig) -> bool:
    """Send a short prompt to pre-load the model into memory.

    The first Ollama call is slow because the model weights must be loaded.
    Warming up avoids a timeout on the first real summary request.
    """
    try:
        logger.info(f"Warming up Ollama model '{config.ollama_model}'...")
        response = httpx.post(
            f"{config.ollama_host}/api/generate",
            json={
                "model": config.ollama_model,
                "prompt": "Say OK.",
                "stream": False,
                "options": {"num_predict": 5},
            },
            timeout=httpx.Timeout(
                connect=config.summary_connect_timeout,
                read=300.0,  # model load can be very slow
                write=config.summary_connect_timeout,
                pool=config.summary_connect_timeout,
            ),
        )
        response.raise_for_status()
        logger.info("Ollama model warm-up complete")
        return True
    except Exception as e:
        logger.warning(f"Ollama warm-up failed: {e}")
        return False


def generate_summary(text: str, config: EnrichmentConfig) -> str:
    """Generate a 2-3 sentence summary of a chunk using Ollama/Llama 3.

    Args:
        text: Chunk text to summarize.
        config: Enrichment configuration with Ollama settings.

    Returns:
        Summary string, or empty string if generation fails.
    """
    if not text.strip():
        return ""

    # Truncate very long texts to avoid overwhelming the model
    truncated = text[:2000] if len(text) > 2000 else text

    prompt = (
        "Summarize the following excerpt from an SEC 10-K filing in 2-3 concise sentences. "
        "Focus on key financial figures, business metrics, and material facts. "
        "Do not add any information not present in the text.\n\n"
        f"Text:\n{truncated}\n\n"
        "Summary:"
    )

    try:
        response = httpx.post(
            f"{config.ollama_host}/api/generate",
            json={
                "model": config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": config.summary_max_tokens},
            },
            timeout=httpx.Timeout(
                connect=config.summary_connect_timeout,
                read=config.summary_timeout,
                write=config.summary_connect_timeout,
                pool=config.summary_connect_timeout,
            ),
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return ""


def enrich_chunks(
    chunks: list[Chunk],
    config: EnrichmentConfig | None = None,
) -> list[EnrichedChunk]:
    """Enrich chunks with keywords, summaries, and metadata.

    Args:
        chunks: Raw chunks from the chunker.
        config: Enrichment configuration.

    Returns:
        List of EnrichedChunks with keywords and summaries.
    """
    if config is None:
        config = EnrichmentConfig()

    logger.info(f"Enriching {len(chunks)} chunks...")

    # Step 1: TF-IDF keyword extraction (batch, fast)
    logger.info("Extracting TF-IDF keywords...")
    keywords_map = extract_keywords(chunks, top_k=config.top_k_keywords)

    # Step 2: Generate summaries (sequential, requires LLM)
    enriched: list[EnrichedChunk] = []
    summary_successes = 0
    summary_failures = 0

    # Warm up the model before starting the batch
    if config.generate_summaries:
        model_ready = _warm_up_model(config)
        if not model_ready:
            logger.warning(
                "Ollama model warm-up failed — summaries may time out. "
                "Is Ollama running? Try: ollama serve"
            )

    for i, chunk in enumerate(chunks):
        kw_pairs = keywords_map.get(chunk.id, [])
        keyword_strings = [kw for kw, _ in kw_pairs]
        keyword_scores = {kw: score for kw, score in kw_pairs}

        # Generate summary if enabled
        summary = ""
        if config.generate_summaries:
            if (i + 1) % 25 == 0 or i == 0:
                logger.info(
                    f"Generating summary {i + 1}/{len(chunks)} "
                    f"({summary_successes} ok, {summary_failures} failed)..."
                )
            summary = generate_summary(chunk.text, config)
            if summary:
                summary_successes += 1
            else:
                summary_failures += 1

        enriched_chunk = EnrichedChunk(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            keywords=keyword_strings,
            summary=summary,
            created_at=chunk.created_at,
            keyword_scores=keyword_scores,
        )
        enriched.append(enriched_chunk)

    logger.info(
        f"Enrichment complete: {len(enriched)} chunks enriched "
        f"({summary_successes} summaries generated, {summary_failures} failed)"
    )
    return enriched
