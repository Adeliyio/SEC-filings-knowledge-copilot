"""Retriever agent — runs hybrid search to find relevant chunks for a query.

Wraps the existing hybrid search pipeline with an agent-friendly interface
that returns structured context for the generator.
"""

import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.storage.search import SearchResult, hybrid_search

logger = logging.getLogger(__name__)

# Map short/common names to the full legal names stored in the database
COMPANY_NAME_MAP: dict[str, str] = {
    "apple": "Apple Inc.",
    "apple inc": "Apple Inc.",
    "apple inc.": "Apple Inc.",
    "aapl": "Apple Inc.",
    "meta": "Meta Platforms, Inc.",
    "meta platforms": "Meta Platforms, Inc.",
    "meta platforms, inc.": "Meta Platforms, Inc.",
    "facebook": "Meta Platforms, Inc.",
    "microsoft": "Microsoft Corporation",
    "microsoft corp": "Microsoft Corporation",
    "microsoft corporation": "Microsoft Corporation",
    "msft": "Microsoft Corporation",
}


def _normalize_company_filter(company_filter: str | None) -> str | None:
    """Normalize a company filter to the full legal name used in the database."""
    if not company_filter:
        return None
    return COMPANY_NAME_MAP.get(company_filter.lower().strip(), company_filter)


@dataclass
class RetrievedContext:
    """Context package returned by the retriever for the generator."""

    query: str
    chunks: list[SearchResult] = field(default_factory=list)
    company_filter: str | None = None

    @property
    def context_text(self) -> str:
        """Format retrieved chunks as numbered context for the LLM."""
        if not self.chunks:
            return "No relevant documents found."

        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            meta = chunk.metadata
            source = meta.get("source_file", "unknown")
            company = meta.get("company_name", "unknown")
            section = meta.get("section_path", "")
            page = meta.get("page_number", "")

            header = f"[Source {i}] {company} — {source}"
            if section:
                header += f" — {section}"
            if page:
                header += f" (p. {page})"

            parts.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(parts)

    @property
    def source_citations(self) -> list[dict]:
        """Extract citation metadata for each chunk."""
        citations = []
        for i, chunk in enumerate(self.chunks, 1):
            meta = chunk.metadata
            citations.append({
                "index": i,
                "chunk_id": chunk.chunk_id,
                "source_file": meta.get("source_file", ""),
                "company_name": meta.get("company_name", ""),
                "section_path": meta.get("section_path", ""),
                "page_number": meta.get("page_number"),
                "is_table": meta.get("is_table", False),
                "relevance_score": chunk.score,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            })
        return citations


def retrieve(
    query: str,
    qdrant_client: QdrantClient,
    db_session: Session,
    top_k: int = 8,
    company_filter: str | None = None,
    use_reranker: bool = True,
) -> RetrievedContext:
    """Run hybrid retrieval for a query.

    Args:
        query: User's search query or sub-query.
        qdrant_client: Qdrant client.
        db_session: PostgreSQL session.
        top_k: Number of chunks to return.
        company_filter: Optional company scope.
        use_reranker: Whether to apply cross-encoder re-ranking.

    Returns:
        RetrievedContext with ranked chunks and formatted context.
    """
    company_filter = _normalize_company_filter(company_filter)
    logger.info(f"Retrieving for query: '{query}' (company={company_filter}, top_k={top_k})")

    results = hybrid_search(
        query=query,
        qdrant_client=qdrant_client,
        db_session=db_session,
        top_k=top_k,
        company_filter=company_filter,
        use_reranker=use_reranker,
    )

    logger.info(f"Retrieved {len(results)} chunks")
    return RetrievedContext(
        query=query,
        chunks=results,
        company_filter=company_filter,
    )
