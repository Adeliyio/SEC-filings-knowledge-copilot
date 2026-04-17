from datetime import datetime

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source_file: str
    company_name: str
    filing_type: str = "10-K"
    fiscal_year: int | None = None
    file_format: str  # "pdf", "htm"
    file_size_bytes: int = 0
    total_pages: int | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    id: str
    metadata: DocumentMetadata
    raw_text: str = ""
    sections: list["Section"] = []


class Section(BaseModel):
    title: str
    level: int  # heading depth (1 = Item, 2 = sub-section, etc.)
    content: str
    page_start: int | None = None
    page_end: int | None = None
    tables: list["TableBlock"] = []


class TableBlock(BaseModel):
    raw_text: str
    markdown: str
    rows: int = 0
    cols: int = 0
    page: int | None = None
