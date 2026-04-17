"""Admin API — re-index triggers, ingestion status, system health."""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.config import settings
from app.storage.postgres import (
    ChunkRow,
    DocumentRow,
    EvalScoreRow,
    FeedbackRow,
    ProvenanceRow,
    get_sync_session_factory,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Track ingestion state across requests
_ingestion_state = {
    "status": "idle",  # idle, running, completed, failed
    "started_at": None,
    "completed_at": None,
    "progress": "",
    "files_processed": 0,
    "total_files": 0,
    "error": None,
}
_ingestion_lock = threading.Lock()


class SystemStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_table_chunks: int
    total_feedback: int
    total_eval_scores: int
    total_provenance: int
    companies: list[str]
    avg_chunks_per_doc: float


class IngestionStatus(BaseModel):
    status: str
    started_at: str | None
    completed_at: str | None
    progress: str
    files_processed: int
    total_files: int
    error: str | None


class ReindexResponse(BaseModel):
    status: str
    message: str


# --- Endpoints ---


@router.get("/stats", response_model=SystemStats)
def get_system_stats() -> SystemStats:
    """Get overall system statistics."""
    factory = get_sync_session_factory()
    session = factory()

    try:
        total_docs = session.query(DocumentRow).count()
        total_chunks = session.query(ChunkRow).count()
        total_table_chunks = (
            session.query(ChunkRow).filter(ChunkRow.is_table == True).count()
        )
        total_feedback = session.query(FeedbackRow).count()
        total_eval_scores = session.query(EvalScoreRow).count()
        total_provenance = session.query(ProvenanceRow).count()

        companies = [
            row[0]
            for row in session.query(DocumentRow.company_name)
            .distinct()
            .all()
        ]

        avg_chunks = total_chunks / max(total_docs, 1)

        return SystemStats(
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_table_chunks=total_table_chunks,
            total_feedback=total_feedback,
            total_eval_scores=total_eval_scores,
            total_provenance=total_provenance,
            companies=sorted(companies),
            avg_chunks_per_doc=round(avg_chunks, 1),
        )
    finally:
        session.close()


@router.get("/ingestion/status", response_model=IngestionStatus)
def ingestion_status() -> IngestionStatus:
    """Get current ingestion pipeline status."""
    with _ingestion_lock:
        return IngestionStatus(
            status=_ingestion_state["status"],
            started_at=_ingestion_state["started_at"],
            completed_at=_ingestion_state["completed_at"],
            progress=_ingestion_state["progress"],
            files_processed=_ingestion_state["files_processed"],
            total_files=_ingestion_state["total_files"],
            error=_ingestion_state["error"],
        )


@router.post("/reindex", response_model=ReindexResponse)
def trigger_reindex(
    file: str | None = Query(None, description="Specific filename to re-index"),
) -> ReindexResponse:
    """Trigger re-ingestion of documents.

    Runs the ingestion pipeline in a background thread.
    Check /api/admin/ingestion/status for progress.
    """
    with _ingestion_lock:
        if _ingestion_state["status"] == "running":
            return ReindexResponse(
                status="already_running",
                message="Ingestion is already in progress.",
            )

    # Start ingestion in background thread
    thread = threading.Thread(
        target=_run_ingestion,
        args=(file,),
        daemon=True,
    )
    thread.start()

    return ReindexResponse(
        status="started",
        message=f"Ingestion started{'for ' + file if file else ' for all files'}.",
    )


def _run_ingestion(target_file: str | None = None) -> None:
    """Background ingestion worker."""
    from app.ingestion.pipeline import ingest_file, store_to_databases
    from app.ingestion.chunker import ChunkerConfig
    from app.ingestion.enrichment import EnrichmentConfig

    with _ingestion_lock:
        _ingestion_state["status"] = "running"
        _ingestion_state["started_at"] = datetime.utcnow().isoformat()
        _ingestion_state["completed_at"] = None
        _ingestion_state["progress"] = "Starting..."
        _ingestion_state["files_processed"] = 0
        _ingestion_state["error"] = None

    try:
        data_dir = settings.data_dir
        supported_exts = {".pdf", ".htm", ".html"}

        if target_file:
            files = [data_dir / target_file]
        else:
            files = sorted(
                f for f in data_dir.iterdir()
                if f.is_file() and f.suffix.lower() in supported_exts
            )

        with _ingestion_lock:
            _ingestion_state["total_files"] = len(files)

        chunker_config = ChunkerConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        enrichment_config = EnrichmentConfig(
            ollama_host=settings.ollama_host,
            ollama_model=settings.ollama_model,
            generate_summaries=True,
        )

        for i, file_path in enumerate(files):
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            with _ingestion_lock:
                _ingestion_state["progress"] = f"Processing {file_path.name}..."

            document, enriched = ingest_file(
                file_path, chunker_config, enrichment_config
            )
            store_to_databases(document, enriched)

            with _ingestion_lock:
                _ingestion_state["files_processed"] = i + 1

        with _ingestion_lock:
            _ingestion_state["status"] = "completed"
            _ingestion_state["completed_at"] = datetime.utcnow().isoformat()
            _ingestion_state["progress"] = "Done"

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        with _ingestion_lock:
            _ingestion_state["status"] = "failed"
            _ingestion_state["error"] = str(e)
            _ingestion_state["completed_at"] = datetime.utcnow().isoformat()


@router.get("/data-files")
def list_data_files() -> dict:
    """List files available in the data directory for ingestion."""
    data_dir = settings.data_dir
    supported_exts = {".pdf", ".htm", ".html"}

    files = []
    if data_dir.exists():
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in supported_exts:
                files.append({
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "format": f.suffix.lower(),
                })

    return {"data_dir": str(data_dir), "files": files}


@router.get("/health")
def admin_health() -> dict:
    """Extended health check with service connectivity."""
    health = {"status": "ok", "services": {}}

    # Check PostgreSQL
    try:
        from sqlalchemy import text
        factory = get_sync_session_factory()
        session = factory()
        try:
            session.execute(text("SELECT 1"))
            health["services"]["postgres"] = "connected"
        finally:
            session.close()
    except Exception as e:
        health["services"]["postgres"] = f"error: {e}"
        health["status"] = "degraded"

    # Check Qdrant
    try:
        from app.storage.qdrant import get_client
        client = get_client()
        collections = client.get_collections()
        health["services"]["qdrant"] = "connected"
        health["services"]["qdrant_collections"] = len(collections.collections)
    except Exception as e:
        health["services"]["qdrant"] = f"error: {e}"
        health["status"] = "degraded"

    # Check Ollama
    try:
        import httpx
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        health["services"]["ollama"] = "connected"
        health["services"]["ollama_models"] = models
    except Exception as e:
        health["services"]["ollama"] = f"error: {e}"
        health["status"] = "degraded"

    return health
