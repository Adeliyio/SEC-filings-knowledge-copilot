"""Feedback API — thumbs up/down on responses."""

import logging
import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from app.storage.postgres import get_sync_session_factory, store_feedback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    query: str
    rating: int  # 1 = thumbs down, 5 = thumbs up
    comment: str = ""


class FeedbackResponse(BaseModel):
    id: str
    status: str


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit user feedback (thumbs up/down) for a response."""
    feedback_id = str(uuid.uuid4())

    factory = get_sync_session_factory()
    session = factory()
    try:
        store_feedback(session, {
            "id": feedback_id,
            "session_id": request.session_id,
            "query": request.query,
            "response_id": request.message_id,
            "rating": request.rating,
            "comment": request.comment,
        })
        session.commit()
        logger.info(f"Feedback stored: rating={request.rating} for message={request.message_id}")
    finally:
        session.close()

    return FeedbackResponse(id=feedback_id, status="ok")
