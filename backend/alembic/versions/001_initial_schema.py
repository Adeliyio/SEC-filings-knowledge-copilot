"""Initial schema: documents, chunks, provenance, feedback, eval_scores, sessions, messages.

Revision ID: 001
Revises: None
Create Date: 2026-04-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- documents ---
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source_file", sa.String(), nullable=False, index=True),
        sa.Column("company_name", sa.String(), nullable=False, index=True),
        sa.Column("filing_type", sa.String(), nullable=False, server_default="10-K"),
        sa.Column("fiscal_year", sa.Integer(), nullable=True),
        sa.Column("file_format", sa.String(), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), server_default="0"),
        sa.Column("total_pages", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # --- chunks ---
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), server_default=""),
        sa.Column("keywords", ARRAY(sa.String()), server_default="{}"),
        sa.Column("keyword_scores", JSONB(), server_default="{}"),
        sa.Column("source_file", sa.String(), nullable=False),
        sa.Column("company_name", sa.String(), nullable=False, index=True),
        sa.Column("filing_type", sa.String(), server_default="10-K"),
        sa.Column("fiscal_year", sa.Integer(), nullable=True),
        sa.Column("section_path", sa.String(), server_default=""),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), server_default="0"),
        sa.Column("is_table", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # --- provenance ---
    op.create_table(
        "provenance",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "chunk_id",
            sa.String(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("details", JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # --- sessions ---
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # --- messages ---
    op.create_table(
        "messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", ARRAY(sa.String()), server_default="{}"),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # --- feedback ---
    op.create_table(
        "feedback",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(),
            sa.ForeignKey("sessions.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("response_id", sa.String(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), server_default=""),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # --- eval_scores ---
    op.create_table(
        "eval_scores",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("response_id", sa.String(), nullable=False, index=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("faithfulness", sa.Float(), nullable=True),
        sa.Column("answer_relevancy", sa.Float(), nullable=True),
        sa.Column("context_precision", sa.Float(), nullable=True),
        sa.Column("context_recall", sa.Float(), nullable=True),
        sa.Column("factual_grounding", sa.Float(), nullable=True),
        sa.Column("completeness", sa.Float(), nullable=True),
        sa.Column("citation_quality", sa.Float(), nullable=True),
        sa.Column("coherence", sa.Float(), nullable=True),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("details", JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("eval_scores")
    op.drop_table("feedback")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("provenance")
    op.drop_table("chunks")
    op.drop_table("documents")
