"""Chat API — streaming SSE endpoint for the multi-agent pipeline."""

import json
import logging
import uuid

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.graph import AgentResponse, run_query, run_query_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


# --- Request/Response Models ---


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class SourceInfo(BaseModel):
    index: int
    chunk_id: str
    source_file: str
    company_name: str
    section_path: str
    page_number: int | None
    is_table: bool
    relevance_score: float
    text_preview: str


class ClaimInfo(BaseModel):
    claim: str
    status: str  # "supported", "partially_supported", "unsupported"
    evidence: str
    reasoning: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[SourceInfo]
    claims: list[ClaimInfo]
    session_id: str
    message_id: str
    attempts: int


# --- Endpoints ---


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint. Runs the full agent pipeline.

    Returns the complete response with confidence scores,
    source citations, and claim verifications.
    """
    logger.info(f"Chat query: '{request.query}'")

    result: AgentResponse = run_query(
        query=request.query,
        session_id=request.session_id,
    )

    sources = [
        SourceInfo(
            index=s.get("index", 0),
            chunk_id=s.get("chunk_id", ""),
            source_file=s.get("source_file", ""),
            company_name=s.get("company_name", ""),
            section_path=s.get("section_path", ""),
            page_number=s.get("page_number"),
            is_table=s.get("is_table", False),
            relevance_score=s.get("relevance_score", 0.0),
            text_preview=s.get("text_preview", ""),
        )
        for s in result.sources
    ]

    claims = [
        ClaimInfo(
            claim=c.get("claim", ""),
            status=c.get("status", ""),
            evidence=c.get("evidence", ""),
            reasoning=c.get("reasoning", ""),
        )
        for c in result.claims
    ]

    return ChatResponse(
        answer=result.answer,
        confidence=result.confidence,
        sources=sources,
        claims=claims,
        session_id=result.session_id,
        message_id=result.message_id,
        attempts=result.attempts,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events.

    Streams tokens as they're generated, then sends source citations,
    claim verifications, and confidence scores at the end.

    Event types:
    - status: Phase updates (planning, retrieving, generating, verifying)
    - plan: Query plan with strategy and sub-queries
    - sources: Retrieved source citations
    - token: Individual generated tokens
    - verification: Claim verification results and confidence
    - retry_answer: New answer from retry attempt
    - done: Final complete response
    """
    logger.info(f"Streaming chat query: '{request.query}'")

    async def event_generator():
        async for event in run_query_stream(
            query=request.query,
            session_id=request.session_id,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
