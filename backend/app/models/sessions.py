from datetime import datetime

from pydantic import BaseModel, Field


class Session(BaseModel):
    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list["Message"] = []


class Message(BaseModel):
    id: str
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    sources: list[str] = []  # chunk IDs used
    confidence: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
