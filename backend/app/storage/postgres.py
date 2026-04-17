"""PostgreSQL storage layer using SQLAlchemy ORM.

Tables: documents, chunks, provenance, feedback, eval_scores, sessions, messages.
"""

import logging
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


# --- Base ---


class Base(DeclarativeBase):
    pass


# --- ORM Models ---


class DocumentRow(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    source_file = Column(String, nullable=False, index=True)
    company_name = Column(String, nullable=False, index=True)
    filing_type = Column(String, nullable=False, default="10-K")
    fiscal_year = Column(Integer, nullable=True)
    file_format = Column(String, nullable=False)
    file_size_bytes = Column(Integer, default=0)
    total_pages = Column(Integer, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("ChunkRow", back_populates="document", cascade="all, delete-orphan")
    provenance = relationship("ProvenanceRow", back_populates="document", cascade="all, delete-orphan")


class ChunkRow(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    summary = Column(Text, default="")
    keywords = Column(ARRAY(String), default=[])
    keyword_scores = Column(JSONB, default={})

    # Metadata (denormalized for fast search filtering)
    source_file = Column(String, nullable=False)
    company_name = Column(String, nullable=False, index=True)
    filing_type = Column(String, default="10-K")
    fiscal_year = Column(Integer, nullable=True)
    section_path = Column(String, default="")
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, default=0)
    is_table = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("DocumentRow", back_populates="chunks")


class ProvenanceRow(Base):
    """Data lineage: tracks the full chain from raw file → parsed → chunked → embedded."""
    __tablename__ = "provenance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(String, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True, index=True)
    stage = Column(String, nullable=False)  # "parsed", "chunked", "enriched", "embedded", "stored"
    details = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("DocumentRow", back_populates="provenance")


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    query = Column(Text, nullable=False)
    response_id = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class EvalScoreRow(Base):
    __tablename__ = "eval_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    response_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=True)

    # RAGAs-style metrics
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)

    # LLM-as-judge scores
    factual_grounding = Column(Float, nullable=True)
    completeness = Column(Float, nullable=True)
    citation_quality = Column(Float, nullable=True)
    coherence = Column(Float, nullable=True)

    # Aggregated
    overall_score = Column(Float, nullable=True)
    details = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class SessionRow(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("MessageRow", back_populates="session", cascade="all, delete-orphan")
    feedback = relationship("FeedbackRow", cascade="all, delete-orphan")


class MessageRow(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(ARRAY(String), default=[])
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionRow", back_populates="messages")


# --- Engine & Session Factories (singletons to avoid connection pool exhaustion) ---

_sync_engine = None
_async_engine = None
_sync_session_factory = None
_async_session_factory = None


def get_sync_engine(url: str | None = None):
    global _sync_engine
    if _sync_engine is None or url is not None:
        engine = create_engine(
            url or settings.postgres_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        if url is not None:
            return engine
        _sync_engine = engine
    return _sync_engine


def get_async_engine(url: str | None = None):
    global _async_engine
    if _async_engine is None or url is not None:
        engine = create_async_engine(
            url or settings.async_postgres_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        if url is not None:
            return engine
        _async_engine = engine
    return _async_engine


def get_sync_session_factory(url: str | None = None) -> sessionmaker[Session]:
    global _sync_session_factory
    if _sync_session_factory is None or url is not None:
        engine = get_sync_engine(url)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        if url is not None:
            return factory
        _sync_session_factory = factory
    return _sync_session_factory


def get_async_session_factory(url: str | None = None) -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None or url is not None:
        engine = get_async_engine(url)
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        if url is not None:
            return factory
        _async_session_factory = factory
    return _async_session_factory


def create_tables(url: str | None = None) -> None:
    """Create all tables (for development/testing — prefer Alembic in production)."""
    engine = get_sync_engine(url)
    Base.metadata.create_all(engine)
    logger.info("Created all PostgreSQL tables")


# --- CRUD Helpers ---


def store_document(session: Session, doc_data: dict) -> DocumentRow:
    """Insert or update a document row."""
    row = session.get(DocumentRow, doc_data["id"])
    if row:
        for k, v in doc_data.items():
            setattr(row, k, v)
    else:
        row = DocumentRow(**doc_data)
        session.add(row)
    return row


def store_chunks(session: Session, chunks_data: list[dict]) -> list[ChunkRow]:
    """Insert or update chunk rows (upsert)."""
    rows = []
    for c in chunks_data:
        existing = session.get(ChunkRow, c["id"])
        if existing:
            for k, v in c.items():
                setattr(existing, k, v)
            rows.append(existing)
        else:
            row = ChunkRow(**c)
            session.add(row)
            rows.append(row)
    return rows


def store_provenance(session: Session, entries: list[dict]) -> list[ProvenanceRow]:
    """Bulk insert provenance entries."""
    rows = [ProvenanceRow(**e) for e in entries]
    session.add_all(rows)
    return rows


def ensure_session(session: Session, session_id: str) -> SessionRow:
    """Get or create a session row so FK constraints are satisfied."""
    row = session.get(SessionRow, session_id)
    if not row:
        row = SessionRow(id=session_id)
        session.add(row)
        session.flush()
    return row


def store_feedback(session: Session, feedback_data: dict) -> FeedbackRow:
    """Insert a feedback row (ensures session exists first)."""
    ensure_session(session, feedback_data["session_id"])
    row = FeedbackRow(**feedback_data)
    session.add(row)
    return row


def store_eval_score(session: Session, score_data: dict) -> EvalScoreRow:
    """Insert an eval score row."""
    row = EvalScoreRow(**score_data)
    session.add(row)
    return row
