"""HTM/XBRL parser for SEC inline XBRL filings using BeautifulSoup."""

import logging
import re
import uuid
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from app.models.documents import Document, DocumentMetadata, Section, TableBlock

logger = logging.getLogger(__name__)

# SEC 10-K item heading patterns
ITEM_PATTERN = re.compile(
    r"^\s*(?:PART\s+[IVX]+|Item\s+\d+[A-Z]?)\b[.:\s—\-]*(.*)",
    re.IGNORECASE,
)

# XBRL namespaced tag prefixes to strip (keep their text content)
XBRL_TAG_PREFIXES = ("ix:", "xbrli:", "link:", "xlink:")

# Tags whose text content should be discarded entirely
XBRL_DISCARD_TAGS = {"ix:hidden", "ix:header", "ix:references", "ix:resources"}


def _strip_xbrl_tags(soup: BeautifulSoup) -> None:
    """Unwrap inline XBRL tags, keeping their text content.

    Tags in XBRL_DISCARD_TAGS are removed entirely (including content).
    All other ix: tags are unwrapped so their text flows into the parent.
    """
    # First remove tags whose content we don't want
    for tag_name in XBRL_DISCARD_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Unwrap remaining XBRL tags (keep their text)
    for prefix in XBRL_TAG_PREFIXES:
        for tag in soup.find_all(re.compile(f"^{re.escape(prefix)}")):
            tag.unwrap()


def _extract_table(table_tag: Tag) -> TableBlock | None:
    """Convert an HTML table to a TableBlock with markdown representation."""
    rows = table_tag.find_all("tr")
    if not rows:
        return None

    parsed_rows: list[list[str]] = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        parsed_rows.append([cell.get_text(strip=True) for cell in cells])

    if not parsed_rows:
        return None

    # Build markdown table
    max_cols = max(len(r) for r in parsed_rows)
    # Pad rows to uniform width
    for r in parsed_rows:
        while len(r) < max_cols:
            r.append("")

    header = parsed_rows[0]
    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in parsed_rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")

    markdown = "\n".join(md_lines)
    raw_text = "\n".join("\t".join(r) for r in parsed_rows)

    return TableBlock(
        raw_text=raw_text,
        markdown=markdown,
        rows=len(parsed_rows),
        cols=max_cols,
    )


def _detect_heading(text: str) -> tuple[int, str] | None:
    """Detect if text is a section heading. Returns (level, title) or None."""
    stripped = text.strip()
    if not stripped:
        return None
    if re.match(r"^PART\s+[IVX]+", stripped, re.IGNORECASE):
        return (1, stripped)
    if re.match(r"^Item\s+\d+[A-Z]?", stripped, re.IGNORECASE):
        return (2, stripped)
    return None


def _is_heading_tag(tag: Tag) -> bool:
    """Check if a tag is styled as a heading (bold, large font, etc.)."""
    if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return True
    style = tag.get("style", "")
    if "font-weight" in style and ("bold" in style or "700" in style or "800" in style or "900" in style):
        text = tag.get_text(strip=True)
        if text and len(text) < 200 and _detect_heading(text):
            return True
    # Check for bold tags wrapping heading text
    if tag.name in ("b", "strong"):
        text = tag.get_text(strip=True)
        if text and _detect_heading(text):
            return True
    return False


def _infer_company_name(text: str, filename: str) -> str:
    """Infer company name from text or filename."""
    # Known companies from our dataset (reliable mapping)
    name_key = filename.split("-")[0].lower()
    company_map = {
        "apple": "Apple Inc.",
        "meta": "Meta Platforms, Inc.",
        "msft": "Microsoft Corporation",
    }
    if name_key in company_map:
        return company_map[name_key]

    # Try SEC cover page pattern
    match = re.search(
        r"(?:Commission\s+[Ff]ile\s+[Nn]umber.*?\n\s*)([A-Z][A-Z\s,\.&]+(?:INC|CORP|LLC|LTD|CO|PLATFORMS)\.?)",
        text[:5000],
    )
    if match:
        return match.group(1).strip()

    return name_key.capitalize()


def _infer_fiscal_year(text: str) -> int | None:
    """Infer fiscal year from document text."""
    match = re.search(
        r"(?:fiscal\s+year\s+ended|for\s+the\s+(?:fiscal\s+)?year\s+ended)"
        r".*?(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r".*?(\d{4})",
        text[:8000],
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    match = re.search(r"10-K.*?(\d{4})", text[:5000])
    if match:
        return int(match.group(1))
    return None


def _walk_body(body: Tag) -> list[Section]:
    """Walk the HTML body and build sections based on heading detection."""
    sections: list[Section] = []
    current_section = Section(title="Preamble", level=0, content="", tables=[])

    for element in body.descendants:
        if isinstance(element, Tag):
            # Check for tables
            if element.name == "table":
                table_block = _extract_table(element)
                if table_block and table_block.rows > 1:
                    current_section.tables.append(table_block)
                continue

            # Check for headings
            if _is_heading_tag(element):
                text = element.get_text(strip=True)
                heading = _detect_heading(text)
                if heading:
                    level, title = heading
                    # Save current section
                    current_section.content = current_section.content.strip()
                    if current_section.content or current_section.tables:
                        sections.append(current_section)
                    current_section = Section(
                        title=title, level=level, content="", tables=[]
                    )
                    continue

        elif isinstance(element, NavigableString):
            text = str(element).strip()
            if text and element.parent.name not in ("table", "tr", "td", "th", "script", "style"):
                # Check if this text is a heading on its own
                heading = _detect_heading(text)
                if heading and len(text) < 200:
                    level, title = heading
                    current_section.content = current_section.content.strip()
                    if current_section.content or current_section.tables:
                        sections.append(current_section)
                    current_section = Section(
                        title=title, level=level, content="", tables=[]
                    )
                else:
                    current_section.content += text + " "

    # Final section
    current_section.content = current_section.content.strip()
    if current_section.content or current_section.tables:
        sections.append(current_section)

    return sections


def parse_htm(file_path: Path) -> Document:
    """Parse an SEC HTM/XBRL filing and return a structured Document.

    Args:
        file_path: Path to the HTM file.

    Returns:
        Document with metadata, raw text, and detected sections/tables.
    """
    logger.info(f"Parsing HTM: {file_path.name}")

    raw_html = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    # Strip XBRL inline tags
    _strip_xbrl_tags(soup)

    # Remove script/style tags
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    body = soup.find("body")
    if not body:
        body = soup

    # Extract full text
    full_text = body.get_text(separator="\n", strip=True)

    # Build sections
    sections = _walk_body(body)

    company_name = _infer_company_name(full_text, file_path.stem)
    fiscal_year = _infer_fiscal_year(full_text)

    metadata = DocumentMetadata(
        source_file=file_path.name,
        company_name=company_name,
        file_format="htm",
        file_size_bytes=file_path.stat().st_size,
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
        f"company={company_name}"
    )
    return document
