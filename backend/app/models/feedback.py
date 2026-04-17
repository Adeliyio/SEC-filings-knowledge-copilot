from datetime import datetime

from pydantic import BaseModel, Field


class Feedback(BaseModel):
    id: str
    session_id: str
    query: str
    response_id: str
    rating: int  # 1 = thumbs down, 5 = thumbs up
    comment: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
