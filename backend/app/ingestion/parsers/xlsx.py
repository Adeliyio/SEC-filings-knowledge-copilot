"""XLSX parser stub — extensibility placeholder for future spreadsheet ingestion."""

from pathlib import Path

from app.models.documents import Document


def parse_xlsx(file_path: Path) -> Document:
    """Parse an XLSX file. Not implemented — stub for extensibility."""
    raise NotImplementedError(
        f"XLSX parsing not yet implemented. File: {file_path.name}"
    )
