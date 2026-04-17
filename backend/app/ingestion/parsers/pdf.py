"""PDF parser using PyMuPDF for text extraction with table detection."""

import logging
import re
import uuid
from pathlib import Path

import pymupdf

from app.models.documents import Document, DocumentMetadata, Section, TableBlock

logger = logging.getLogger(__name__)

# SEC 10-K Item headings pattern
ITEM_PATTERN = re.compile(
    r"^(?:PART\s+[IVX]+|Item\s+\d+[A-Z]?)\b[.:\s—\-]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Table heuristic: lines with 3+ columns separated by whitespace/tabs
TABLE_LINE_PATTERN = re.compile(r"(?:\S+\s{2,}){2,}\S+")


def _detect_heading_level(text: str) -> int | None:
    """Detect heading level from text patterns in SEC filings."""
    stripped = text.strip()
    if re.match(r"^PART\s+[IVX]+", stripped, re.IGNORECASE):
        return 1
    if re.match(r"^Item\s+\d+[A-Z]?", stripped, re.IGNORECASE):
        return 2
    return None


def _is_table_block(text: str) -> bool:
    """Heuristic: a block is a table if most lines have aligned columns."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    table_lines = sum(1 for l in lines if TABLE_LINE_PATTERN.search(l))
    return table_lines / len(lines) > 0.5


def _extract_tables_from_page(page: pymupdf.Page) -> list[TableBlock]:
    """Extract tables from a page using PyMuPDF's table finder."""
    tables = []
    try:
        tab_finder = page.find_tables()
        for table in tab_finder.tables:
            df = table.to_pandas()
            markdown = df.to_markdown(index=False)
            raw_text = df.to_string(index=False)
            tables.append(
                TableBlock(
                    raw_text=raw_text,
                    markdown=markdown,
                    rows=len(df),
                    cols=len(df.columns),
                    page=page.number + 1,
                )
            )
    except Exception as e:
        logger.debug(f"Table extraction failed on page {page.number + 1}: {e}")
    return tables


def _build_sections(pages_text: list[dict]) -> list[Section]:
    """Build sections from extracted page text by detecting headings."""
    sections: list[Section] = []
    current_section = Section(
        title="Preamble",
        level=0,
        content="",
        page_start=1,
        tables=[],
    )

    for page_info in pages_text:
        page_num = page_info["page"]
        text = page_info["text"]
        page_tables = page_info["tables"]

        for line in text.split("\n"):
            heading_level = _detect_heading_level(line)
            if heading_level is not None:
                # Save current section
                current_section.content = current_section.content.strip()
                current_section.page_end = page_num
                if current_section.content or current_section.tables:
                    sections.append(current_section)

                # Start new section
                title = line.strip()
                current_section = Section(
                    title=title,
                    level=heading_level,
                    content="",
                    page_start=page_num,
                    tables=[],
                )
            else:
                current_section.content += line + "\n"

        current_section.tables.extend(page_tables)

    # Don't forget last section
    current_section.content = current_section.content.strip()
    current_section.page_end = pages_text[-1]["page"] if pages_text else None
    if current_section.content or current_section.tables:
        sections.append(current_section)

    return sections


def _infer_company_name(text: str, filename: str) -> str:
    """Infer company name from the document text or filename."""
    # Known companies from our dataset (reliable mapping)
    name_key = filename.split("-")[0].lower()
    company_map = {
        "apple": "Apple Inc.",
        "meta": "Meta Platforms, Inc.",
        "msft": "Microsoft Corporation",
    }
    if name_key in company_map:
        return company_map[name_key]

    # Try SEC cover page pattern: "COMPANY NAME" on its own line before "FORM 10-K"
    match = re.search(
        r"(?:Commission\s+[Ff]ile\s+[Nn]umber.*?\n\s*)([A-Z][A-Z\s,\.&]+(?:INC|CORP|LLC|LTD|CO|PLATFORMS)\.?)",
        text[:5000],
    )
    if match:
        return match.group(1).strip()

    return name_key.capitalize()


def _infer_fiscal_year(text: str) -> int | None:
    """Infer fiscal year from early pages of the document."""
    match = re.search(
        r"(?:fiscal\s+year\s+ended|for\s+the\s+(?:fiscal\s+)?year\s+ended)"
        r".*?(?:september|october|november|december|june|july).*?(\d{4})",
        text[:5000],
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    # Fallback: look for a 4-digit year near "10-K"
    match = re.search(r"10-K.*?(\d{4})", text[:3000])
    if match:
        return int(match.group(1))
    return None


def parse_pdf(file_path: Path) -> Document:
    """Parse a PDF file and return a structured Document.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Document with metadata, raw text, and detected sections/tables.
    """
    logger.info(f"Parsing PDF: {file_path.name}")
    doc = pymupdf.open(str(file_path))

    pages_text = []
    full_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        tables = _extract_tables_from_page(page)

        pages_text.append({
            "page": page_num + 1,
            "text": text,
            "tables": tables,
        })
        full_text_parts.append(text)

    full_text = "\n".join(full_text_parts)
    sections = _build_sections(pages_text)

    company_name = _infer_company_name(full_text, file_path.stem)
    fiscal_year = _infer_fiscal_year(full_text)

    page_count = len(doc)
    doc.close()

    metadata = DocumentMetadata(
        source_file=file_path.name,
        company_name=company_name,
        file_format="pdf",
        file_size_bytes=file_path.stat().st_size,
        total_pages=page_count,
        fiscal_year=fiscal_year,
    )

    document = Document(
        id=str(uuid.uuid4()),
        metadata=metadata,
        raw_text=full_text,
        sections=sections,
    )

    logger.info(
        f"Parsed {file_path.name}: {len(sections)} sections, "
        f"{page_count} pages, company={company_name}"
    )
    return document
