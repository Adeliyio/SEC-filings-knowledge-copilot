"""Structure-aware chunking for SEC filings.

Three components run in sequence:
1. Table preserver — keeps financial table rows intact
2. Heading detector — uses heading hierarchy to define section boundaries
3. Boundary detector — identifies semantic breaks (paragraphs, footnotes, list items)
"""

import logging
import re
import uuid
from dataclasses import dataclass, field

from app.models.chunks import Chunk, ChunkMetadata
from app.models.documents import Document, Section, TableBlock

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_TABLE_CHUNK_SIZE = 3000  # tables get more room


@dataclass
class ChunkerConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_chunk_size: int = MIN_CHUNK_SIZE
    max_table_chunk_size: int = MAX_TABLE_CHUNK_SIZE


# --- Component 1: Table Preserver ---


def _create_table_chunks(
    table: TableBlock,
    document: Document,
    section: Section,
    chunk_index: int,
    config: ChunkerConfig,
) -> list[Chunk]:
    """Create chunks from a table, keeping rows intact.

    If the table fits in one chunk, return it as-is.
    Otherwise, split at row boundaries, never mid-row.
    """
    chunks = []
    table_text = table.markdown

    if len(table_text) <= config.max_table_chunk_size:
        chunk = Chunk(
            id=str(uuid.uuid4()),
            text=table_text,
            metadata=ChunkMetadata(
                document_id=document.id,
                source_file=document.metadata.source_file,
                company_name=document.metadata.company_name,
                fiscal_year=document.metadata.fiscal_year,
                section_path=section.title,
                page_number=table.page,
                chunk_index=chunk_index,
                is_table=True,
            ),
        )
        chunks.append(chunk)
    else:
        # Split table at row boundaries
        lines = table_text.split("\n")
        # Keep header (first 2 lines: header + separator) with each chunk
        header_lines = lines[:2] if len(lines) >= 2 else lines[:1]
        header = "\n".join(header_lines)
        data_lines = lines[len(header_lines):]

        current_chunk_lines = []
        current_size = len(header)

        for line in data_lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size > config.max_table_chunk_size and current_chunk_lines:
                chunk_text = header + "\n" + "\n".join(current_chunk_lines)
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        source_file=document.metadata.source_file,
                        company_name=document.metadata.company_name,
                        fiscal_year=document.metadata.fiscal_year,
                        section_path=section.title,
                        page_number=table.page,
                        chunk_index=chunk_index + len(chunks),
                        is_table=True,
                    ),
                ))
                current_chunk_lines = []
                current_size = len(header)

            current_chunk_lines.append(line)
            current_size += line_size

        if current_chunk_lines:
            chunk_text = header + "\n" + "\n".join(current_chunk_lines)
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                metadata=ChunkMetadata(
                    document_id=document.id,
                    source_file=document.metadata.source_file,
                    company_name=document.metadata.company_name,
                    fiscal_year=document.metadata.fiscal_year,
                    section_path=section.title,
                    page_number=table.page,
                    chunk_index=chunk_index + len(chunks),
                    is_table=True,
                ),
            ))

    return chunks


# --- Component 2: Heading Detector ---


def _build_section_path(section: Section, parent_sections: list[Section]) -> str:
    """Build a hierarchical section path like 'PART I > Item 1 > Description of Business'."""
    path_parts = []
    for ps in parent_sections:
        if ps.level < section.level:
            path_parts.append(ps.title)
    path_parts.append(section.title)
    return " > ".join(path_parts)


# --- Component 3: Boundary Detector ---

# Patterns that indicate semantic boundaries
BOUNDARY_PATTERNS = [
    re.compile(r"\n\s*\n"),              # double newline (paragraph break)
    re.compile(r"\n\s*[•●▪\-\*]\s"),    # bullet list item
    re.compile(r"\n\s*\(\d+\)\s"),       # numbered footnote like (1)
    re.compile(r"\n\s*\d+\.\s"),         # numbered list item
    re.compile(r"\n\s*[a-z]\)\s"),       # lettered list item like a)
    re.compile(r"\n\s*Note\s+\d+", re.IGNORECASE),  # footnote heading
]


def _find_best_split_point(text: str, target: int, window: int = 100) -> int:
    """Find the best split point near `target` using semantic boundaries.

    Looks within [target - window, target + window] for a boundary pattern.
    Falls back to the nearest sentence end, then nearest whitespace.
    """
    search_start = max(0, target - window)
    search_end = min(len(text), target + window)
    search_region = text[search_start:search_end]

    # Try semantic boundary patterns
    best_pos = None
    best_distance = window + 1

    for pattern in BOUNDARY_PATTERNS:
        for match in pattern.finditer(search_region):
            pos = search_start + match.start()
            distance = abs(pos - target)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos

    if best_pos is not None:
        return best_pos

    # Fallback: sentence boundary (period + space/newline)
    for i in range(target, min(target + window, len(text) - 1)):
        if text[i] == "." and i + 1 < len(text) and text[i + 1] in " \n\t":
            return i + 1
    for i in range(target, max(target - window, 0), -1):
        if text[i] == "." and i + 1 < len(text) and text[i + 1] in " \n\t":
            return i + 1

    # Last fallback: nearest whitespace
    for i in range(target, min(target + window, len(text))):
        if text[i] in " \n\t":
            return i
    return target


def _chunk_text(
    text: str,
    document: Document,
    section: Section,
    section_path: str,
    start_index: int,
    config: ChunkerConfig,
    page_number: int | None = None,
) -> list[Chunk]:
    """Split section text into overlapping chunks at semantic boundaries."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= config.chunk_size:
        return [Chunk(
            id=str(uuid.uuid4()),
            text=text,
            metadata=ChunkMetadata(
                document_id=document.id,
                source_file=document.metadata.source_file,
                company_name=document.metadata.company_name,
                fiscal_year=document.metadata.fiscal_year,
                section_path=section_path,
                page_number=page_number or section.page_start,
                chunk_index=start_index,
                is_table=False,
            ),
        )]

    chunks = []
    pos = 0
    idx = start_index

    while pos < len(text):
        end = pos + config.chunk_size

        if end >= len(text):
            chunk_text = text[pos:].strip()
        else:
            split_at = _find_best_split_point(text, end)
            chunk_text = text[pos:split_at].strip()
            end = split_at

        if len(chunk_text) >= config.min_chunk_size:
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                metadata=ChunkMetadata(
                    document_id=document.id,
                    source_file=document.metadata.source_file,
                    company_name=document.metadata.company_name,
                    fiscal_year=document.metadata.fiscal_year,
                    section_path=section_path,
                    page_number=page_number or section.page_start,
                    chunk_index=idx,
                    is_table=False,
                ),
            ))
            idx += 1

        # Advance with overlap
        pos = max(pos + 1, end - config.chunk_overlap)

    return chunks


# --- Main Chunking Entrypoint ---


def chunk_document(document: Document, config: ChunkerConfig | None = None) -> list[Chunk]:
    """Chunk a parsed document using structure-aware splitting.

    Pipeline: table preserver → heading detector → boundary detector.

    Args:
        document: Parsed Document with sections and tables.
        config: Chunking configuration. Uses defaults if None.

    Returns:
        List of Chunks with metadata and provenance.
    """
    if config is None:
        config = ChunkerConfig()

    all_chunks: list[Chunk] = []
    chunk_index = 0

    # Track parent sections for hierarchical path building
    parent_sections: list[Section] = []

    for section in document.sections:
        # Update parent section stack
        while parent_sections and parent_sections[-1].level >= section.level:
            parent_sections.pop()

        section_path = _build_section_path(section, parent_sections)
        parent_sections.append(section)

        # Component 1: Table preserver — chunk tables first
        for table in section.tables:
            table_chunks = _create_table_chunks(
                table, document, section, chunk_index, config
            )
            all_chunks.extend(table_chunks)
            chunk_index += len(table_chunks)

        # Components 2+3: Heading detector + boundary detector — chunk text
        if section.content:
            text_chunks = _chunk_text(
                section.content,
                document,
                section,
                section_path,
                chunk_index,
                config,
            )
            all_chunks.extend(text_chunks)
            chunk_index += len(text_chunks)

    logger.info(
        f"Chunked {document.metadata.source_file}: "
        f"{len(all_chunks)} chunks from {len(document.sections)} sections"
    )
    return all_chunks
