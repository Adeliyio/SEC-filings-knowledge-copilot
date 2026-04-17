"""Search API endpoint — returns ranked chunks for a query."""

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.config import settings
from app.storage.postgres import get_sync_session_factory
from app.storage.qdrant import get_client as get_qdrant_client
from app.storage.search import SearchResult, hybrid_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


# --- Response Models ---


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    dense_score: float
    bm25_score: float
    rerank_score: float | None
    source_file: str
    company_name: str
    section_path: str
    page_number: int | None
    is_table: bool
    keywords: list[str]
    summary: str


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total_results: int
    company_filter: str | None


# --- Dependencies ---


def get_db() -> Session:
    factory = get_sync_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def get_qdrant() -> QdrantClient:
    return get_qdrant_client()


# --- Endpoint ---


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    company: str | None = Query(None, description="Filter by company name"),
    use_reranker: bool = Query(True, description="Apply cross-encoder re-ranking"),
    db: Session = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> SearchResponse:
    """Search across SEC 10-K filings using hybrid retrieval.

    Combines dense vector search (semantic) with BM25 keyword matching,
    then re-ranks results with a cross-encoder for precision.
    """
    logger.info(f"Search query: '{q}', top_k={top_k}, company={company}")

    results: list[SearchResult] = hybrid_search(
        query=q,
        qdrant_client=qdrant,
        db_session=db,
        top_k=top_k,
        company_filter=company,
        use_reranker=use_reranker,
    )

    chunk_results = [
        ChunkResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=r.score,
            dense_score=r.dense_score,
            bm25_score=r.bm25_score,
            rerank_score=r.rerank_score,
            source_file=r.metadata.get("source_file", ""),
            company_name=r.metadata.get("company_name", ""),
            section_path=r.metadata.get("section_path", ""),
            page_number=r.metadata.get("page_number"),
            is_table=r.metadata.get("is_table", False),
            keywords=r.metadata.get("keywords", []),
            summary=r.metadata.get("summary", ""),
        )
        for r in results
    ]

    return SearchResponse(
        query=q,
        results=chunk_results,
        total_results=len(chunk_results),
        company_filter=company,
    )
