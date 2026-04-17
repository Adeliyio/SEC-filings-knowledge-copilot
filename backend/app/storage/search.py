"""Hybrid search: dense vector (Qdrant) + BM25 keyword matching + cross-encoder re-ranking.

Combines semantic search with keyword matching for higher recall,
then uses a cross-encoder to re-rank the top candidates for precision.
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from sqlalchemy.orm import Session

from app.storage.embeddings import embed_query
from app.storage.postgres import ChunkRow
from app.storage.qdrant import VectorSearchResult, search_dense

logger = logging.getLogger(__name__)

# Cross-encoder model for re-ranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Singleton to avoid reloading the model on every search
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading cross-encoder model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# --- BM25 Implementation ---


@dataclass
class BM25Index:
    """Simple BM25 index over chunk texts stored in PostgreSQL."""

    doc_freqs: dict[str, int] = field(default_factory=dict)
    doc_lens: dict[str, int] = field(default_factory=dict)
    avg_doc_len: float = 0.0
    total_docs: int = 0
    # term -> {chunk_id -> term_freq}
    inverted_index: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))

    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(db_session: Session, company_filter: str | None = None) -> BM25Index:
    """Build a BM25 index from chunks in PostgreSQL.

    Args:
        db_session: SQLAlchemy session.
        company_filter: Optional company name to scope the index.

    Returns:
        BM25Index ready for querying.
    """
    query = db_session.query(ChunkRow.id, ChunkRow.text, ChunkRow.keywords)
    if company_filter:
        query = query.filter(ChunkRow.company_name == company_filter)

    chunks = query.all()
    index = BM25Index(total_docs=len(chunks))

    if not chunks:
        return index

    total_len = 0
    for chunk_id, text, keywords in chunks:
        # Combine text and keywords for richer term matching
        combined = text + " " + " ".join(keywords or [])
        tokens = _tokenize(combined)
        doc_len = len(tokens)
        index.doc_lens[chunk_id] = doc_len
        total_len += doc_len

        # Count term frequencies
        term_freqs: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1

        for term, freq in term_freqs.items():
            index.inverted_index[term][chunk_id] = freq

    index.avg_doc_len = total_len / max(len(chunks), 1)

    # Document frequencies
    for term, postings in index.inverted_index.items():
        index.doc_freqs[term] = len(postings)

    return index


def bm25_search(index: BM25Index, query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Score documents against a query using BM25.

    Args:
        index: Pre-built BM25 index.
        query: Search query string.
        top_k: Number of results.

    Returns:
        List of (chunk_id, score) tuples, sorted by score descending.
    """
    if index.total_docs == 0:
        return []

    query_tokens = _tokenize(query)
    scores: dict[str, float] = defaultdict(float)

    for token in query_tokens:
        if token not in index.inverted_index:
            continue

        df = index.doc_freqs.get(token, 0)
        idf = math.log((index.total_docs - df + 0.5) / (df + 0.5) + 1.0)

        for chunk_id, tf in index.inverted_index[token].items():
            doc_len = index.doc_lens.get(chunk_id, 0)
            tf_norm = (tf * (index.k1 + 1)) / (
                tf + index.k1 * (1 - index.b + index.b * doc_len / max(index.avg_doc_len, 1))
            )
            scores[chunk_id] += idf * tf_norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# --- Hybrid Search ---


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    dense_score: float
    bm25_score: float
    rerank_score: float | None
    metadata: dict


def hybrid_search(
    query: str,
    qdrant_client: QdrantClient,
    db_session: Session,
    top_k: int = 10,
    dense_top_k: int = 30,
    bm25_top_k: int = 30,
    company_filter: str | None = None,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    use_reranker: bool = True,
) -> list[SearchResult]:
    """Run hybrid search: dense vector + BM25, then cross-encoder re-rank.

    Args:
        query: User's search query.
        qdrant_client: Qdrant client for dense search.
        db_session: PostgreSQL session for BM25.
        top_k: Final number of results to return.
        dense_top_k: Candidates from dense search.
        bm25_top_k: Candidates from BM25 search.
        company_filter: Optional company name scope.
        dense_weight: Weight for dense search scores in fusion.
        bm25_weight: Weight for BM25 scores in fusion.
        use_reranker: Whether to apply cross-encoder re-ranking.

    Returns:
        List of SearchResult, ordered by relevance.
    """
    # Step 1: Dense vector search
    logger.info("Running dense vector search...")
    query_embedding = embed_query(query)
    dense_results = search_dense(
        qdrant_client, query_embedding, top_k=dense_top_k, company_filter=company_filter
    )

    # Step 2: BM25 keyword search
    logger.info("Running BM25 keyword search...")
    bm25_index = build_bm25_index(db_session, company_filter=company_filter)
    bm25_results = bm25_search(bm25_index, query, top_k=bm25_top_k)

    # Step 3: Reciprocal Rank Fusion (RRF) to combine scores
    logger.info("Fusing dense + BM25 results...")
    rrf_k = 60  # Standard RRF constant

    fused_scores: dict[str, dict] = {}

    # Score dense results
    for rank, result in enumerate(dense_results):
        cid = result.chunk_id
        rrf_score = dense_weight / (rrf_k + rank + 1)
        fused_scores[cid] = {
            "dense_score": result.score,
            "bm25_score": 0.0,
            "rrf_score": rrf_score,
            "payload": result.payload,
        }

    # Score BM25 results
    for rank, (cid, bm25_score) in enumerate(bm25_results):
        rrf_score = bm25_weight / (rrf_k + rank + 1)
        if cid in fused_scores:
            fused_scores[cid]["bm25_score"] = bm25_score
            fused_scores[cid]["rrf_score"] += rrf_score
        else:
            fused_scores[cid] = {
                "dense_score": 0.0,
                "bm25_score": bm25_score,
                "rrf_score": rrf_score,
                "payload": {},
            }

    # Sort by fused RRF score
    ranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x]["rrf_score"], reverse=True)

    # Take more candidates than top_k for re-ranking
    rerank_candidates = ranked_ids[: top_k * 3] if use_reranker else ranked_ids[:top_k]

    # Fetch full chunk texts from PostgreSQL
    chunk_rows = (
        db_session.query(ChunkRow)
        .filter(ChunkRow.id.in_(rerank_candidates))
        .all()
    )
    chunk_map = {c.id: c for c in chunk_rows}

    # Step 4: Cross-encoder re-ranking
    results: list[SearchResult] = []

    if use_reranker and rerank_candidates:
        logger.info(f"Re-ranking {len(rerank_candidates)} candidates with cross-encoder...")
        reranker = _get_reranker()

        pairs = []
        valid_ids = []
        for cid in rerank_candidates:
            if cid in chunk_map:
                pairs.append((query, chunk_map[cid].text))
                valid_ids.append(cid)

        if pairs:
            rerank_scores = reranker.predict(pairs)

            scored = list(zip(valid_ids, rerank_scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            for cid, rerank_score in scored[:top_k]:
                chunk = chunk_map[cid]
                info = fused_scores.get(cid, {})
                results.append(SearchResult(
                    chunk_id=cid,
                    text=chunk.text,
                    score=float(rerank_score),
                    dense_score=info.get("dense_score", 0.0),
                    bm25_score=info.get("bm25_score", 0.0),
                    rerank_score=float(rerank_score),
                    metadata={
                        "source_file": chunk.source_file,
                        "company_name": chunk.company_name,
                        "section_path": chunk.section_path,
                        "page_number": chunk.page_number,
                        "is_table": chunk.is_table,
                        "chunk_index": chunk.chunk_index,
                        "keywords": chunk.keywords or [],
                        "summary": chunk.summary or "",
                    },
                ))
    else:
        # Without re-ranker, use RRF scores directly
        for cid in rerank_candidates[:top_k]:
            if cid not in chunk_map:
                continue
            chunk = chunk_map[cid]
            info = fused_scores.get(cid, {})
            results.append(SearchResult(
                chunk_id=cid,
                text=chunk.text,
                score=info.get("rrf_score", 0.0),
                dense_score=info.get("dense_score", 0.0),
                bm25_score=info.get("bm25_score", 0.0),
                rerank_score=None,
                metadata={
                    "source_file": chunk.source_file,
                    "company_name": chunk.company_name,
                    "section_path": chunk.section_path,
                    "page_number": chunk.page_number,
                    "is_table": chunk.is_table,
                    "chunk_index": chunk.chunk_index,
                    "keywords": chunk.keywords or [],
                    "summary": chunk.summary or "",
                },
            ))

    logger.info(f"Hybrid search returned {len(results)} results")
    return results
