from datetime import datetime

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_id: str
    source_file: str
    company_name: str
    filing_type: str = "10-K"
    fiscal_year: int | None = None
    section_path: str = ""  # e.g., "Item 7 > Revenue"
    page_number: int | None = None
    chunk_index: int = 0
    is_table: bool = False


class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    keywords: list[str] = []
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EnrichedChunk(Chunk):
    embedding: list[float] | None = None
    keyword_scores: dict[str, float] = {}
