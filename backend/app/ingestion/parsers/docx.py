"""DOCX parser stub — extensibility placeholder for future document types."""

from pathlib import Path

from app.models.documents import Document


def parse_docx(file_path: Path) -> Document:
    """Parse a DOCX file. Not implemented — stub for extensibility."""
    raise NotImplementedError(
        f"DOCX parsing not yet implemented. File: {file_path.name}"
    )
