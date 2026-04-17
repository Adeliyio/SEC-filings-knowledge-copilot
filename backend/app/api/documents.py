"""Documents API — document listing, upload, and management."""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.storage.postgres import DocumentRow, ChunkRow, get_sync_session_factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".htm", ".html"}


class DocumentInfo(BaseModel):
    id: str
    source_file: str
    company_name: str
    filing_type: str
    fiscal_year: int | None
    file_format: str
    file_size_bytes: int
    total_pages: int | None
    chunk_count: int
    ingested_at: str | None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str


# --- Endpoints ---


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """List all ingested documents with chunk counts."""
    factory = get_sync_session_factory()
    session = factory()

    try:
        docs = session.query(DocumentRow).order_by(DocumentRow.ingested_at.desc()).all()

        doc_infos = []
        for doc in docs:
            chunk_count = (
                session.query(ChunkRow)
                .filter(ChunkRow.document_id == doc.id)
                .count()
            )
            doc_infos.append(DocumentInfo(
                id=doc.id,
                source_file=doc.source_file,
                company_name=doc.company_name,
                filing_type=doc.filing_type,
                fiscal_year=doc.fiscal_year,
                file_format=doc.file_format,
                file_size_bytes=doc.file_size_bytes or 0,
                total_pages=doc.total_pages,
                chunk_count=chunk_count,
                ingested_at=doc.ingested_at.isoformat() if doc.ingested_at else None,
            ))

        return DocumentListResponse(documents=doc_infos, total=len(doc_infos))
    finally:
        session.close()


@router.get("/{document_id}")
def get_document(document_id: str) -> dict:
    """Get detailed info for a single document."""
    factory = get_sync_session_factory()
    session = factory()

    try:
        doc = session.get(DocumentRow, document_id)
        if not doc:
            return {"error": "Document not found"}

        chunks = (
            session.query(ChunkRow)
            .filter(ChunkRow.document_id == document_id)
            .all()
        )

        table_chunks = sum(1 for c in chunks if c.is_table)
        sections = set(c.section_path for c in chunks if c.section_path)

        return {
            "id": doc.id,
            "source_file": doc.source_file,
            "company_name": doc.company_name,
            "filing_type": doc.filing_type,
            "fiscal_year": doc.fiscal_year,
            "file_format": doc.file_format,
            "file_size_bytes": doc.file_size_bytes or 0,
            "total_pages": doc.total_pages,
            "chunk_count": len(chunks),
            "table_chunk_count": table_chunks,
            "sections": sorted(sections),
            "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
        }
    finally:
        session.close()


@router.get("/{document_id}/chunks")
def get_document_chunks(
    document_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Get chunks for a specific document."""
    factory = get_sync_session_factory()
    session = factory()

    try:
        chunks = (
            session.query(ChunkRow)
            .filter(ChunkRow.document_id == document_id)
            .order_by(ChunkRow.chunk_index)
            .offset(offset)
            .limit(limit)
            .all()
        )

        total = (
            session.query(ChunkRow)
            .filter(ChunkRow.document_id == document_id)
            .count()
        )

        return {
            "document_id": document_id,
            "total": total,
            "chunks": [
                {
                    "id": c.id,
                    "chunk_index": c.chunk_index,
                    "section_path": c.section_path,
                    "page_number": c.page_number,
                    "is_table": c.is_table,
                    "text_preview": c.text[:300] if c.text else "",
                    "keywords": c.keywords or [],
                    "summary": c.summary or "",
                }
                for c in chunks
            ],
        }
    finally:
        session.close()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a new document for ingestion.

    Saves the file to the data directory. Actual ingestion is triggered
    separately via the admin re-index endpoint.
    """
    if not file.filename:
        return UploadResponse(
            status="error", filename="", message="No filename provided"
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return UploadResponse(
            status="error",
            filename=file.filename,
            message=f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    dest = settings.data_dir / file.filename
    try:
        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Uploaded document: {file.filename} ({len(content)} bytes)")
        return UploadResponse(
            status="ok",
            filename=file.filename,
            message=f"Uploaded to {dest}. Trigger re-index to ingest.",
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return UploadResponse(
            status="error",
            filename=file.filename or "",
            message=str(e),
        )
