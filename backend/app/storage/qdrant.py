"""Qdrant vector store for chunk embeddings.

Handles collection creation, upserting vectors with payloads,
and dense vector search with metadata filtering.
"""

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.storage.embeddings import EMBEDDING_DIM

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    chunk_id: str
    score: float
    payload: dict


def get_client(host: str | None = None, port: int | None = None) -> QdrantClient:
    """Create a Qdrant client."""
    return QdrantClient(
        host=host or settings.qdrant_host,
        port=port or settings.qdrant_port,
        timeout=60,
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str | None = None,
    vector_size: int = EMBEDDING_DIM,
) -> None:
    """Create the collection if it doesn't exist."""
    name = collection_name or settings.qdrant_collection

    collections = [c.name for c in client.get_collections().collections]
    if name in collections:
        logger.info(f"Collection '{name}' already exists")
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"Created collection '{name}' (dim={vector_size}, cosine)")


def upsert_chunks(
    client: QdrantClient,
    chunk_ids: list[str],
    embeddings: list[list[float]],
    payloads: list[dict],
    collection_name: str | None = None,
    batch_size: int = 100,
) -> int:
    """Upsert chunk vectors with metadata payloads into Qdrant.

    Args:
        client: Qdrant client.
        chunk_ids: List of chunk IDs (used as point IDs via UUID).
        embeddings: Corresponding embedding vectors.
        payloads: Metadata dicts for each chunk (company, section, etc.).
        collection_name: Target collection.
        batch_size: Points per upsert call.

    Returns:
        Number of points upserted.
    """
    name = collection_name or settings.qdrant_collection
    total = 0

    for i in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        batch_payloads = payloads[i : i + batch_size]

        points = [
            PointStruct(
                id=cid,
                vector=emb,
                payload=payload,
            )
            for cid, emb, payload in zip(batch_ids, batch_embeddings, batch_payloads)
        ]

        client.upsert(collection_name=name, points=points)
        total += len(points)
        logger.debug(f"Upserted batch {i // batch_size + 1}: {len(points)} points")

    logger.info(f"Upserted {total} points into '{name}'")
    return total


def search_dense(
    client: QdrantClient,
    query_embedding: list[float],
    top_k: int = 20,
    company_filter: str | None = None,
    collection_name: str | None = None,
) -> list[VectorSearchResult]:
    """Dense vector search against Qdrant.

    Args:
        client: Qdrant client.
        query_embedding: Query vector.
        top_k: Number of results to return.
        company_filter: Optional company name filter.
        collection_name: Target collection.

    Returns:
        Ranked list of VectorSearchResult.
    """
    name = collection_name or settings.qdrant_collection

    query_filter = None
    if company_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="company_name",
                    match=MatchValue(value=company_filter),
                )
            ]
        )

    results = client.query_points(
        collection_name=name,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        VectorSearchResult(
            chunk_id=str(hit.id),
            score=hit.score,
            payload=hit.payload or {},
        )
        for hit in results.points
    ]


def delete_by_document(
    client: QdrantClient,
    document_id: str,
    collection_name: str | None = None,
) -> None:
    """Delete all points for a given document ID."""
    name = collection_name or settings.qdrant_collection
    client.delete(
        collection_name=name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )
    logger.info(f"Deleted points for document {document_id} from '{name}'")


def get_collection_info(
    client: QdrantClient,
    collection_name: str | None = None,
) -> dict:
    """Get collection stats."""
    name = collection_name or settings.qdrant_collection
    info = client.get_collection(name)
    return {
        "name": name,
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
        "status": str(info.status),
    }
