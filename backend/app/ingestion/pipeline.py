"""Ingestion pipeline: orchestrates parsing → chunking → enrichment → storage.

CLI entrypoint that processes all 10-K filings in the data directory
and outputs enriched chunks, optionally storing them in Qdrant + PostgreSQL.
"""

import json
import logging
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table as RichTable

from app.config import settings
from app.ingestion.chunker import ChunkerConfig, chunk_document
from app.ingestion.enrichment import EnrichmentConfig, enrich_chunks
from app.ingestion.parsers.htm import parse_htm
from app.ingestion.parsers.pdf import parse_pdf
from app.models.chunks import EnrichedChunk
from app.models.documents import Document

logger = logging.getLogger(__name__)
console = Console()

# Supported file extensions → parser mapping
PARSERS = {
    ".pdf": parse_pdf,
    ".htm": parse_htm,
    ".html": parse_htm,
}


def parse_file(file_path: Path) -> Document:
    """Parse a file using the appropriate parser based on extension."""
    ext = file_path.suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file format: {ext} ({file_path.name})")
    return parser(file_path)


def ingest_file(
    file_path: Path,
    chunker_config: ChunkerConfig | None = None,
    enrichment_config: EnrichmentConfig | None = None,
) -> tuple[Document, list[EnrichedChunk]]:
    """Run the full ingestion pipeline on a single file.

    Pipeline: parse → chunk → enrich.

    Returns:
        Tuple of (Document, list of enriched chunks).
    """
    # Step 1: Parse
    console.print(f"[bold blue]Parsing[/] {file_path.name}...")
    start = time.time()
    document = parse_file(file_path)
    parse_time = time.time() - start
    console.print(
        f"  Parsed in {parse_time:.1f}s — "
        f"{len(document.sections)} sections, "
        f"company: {document.metadata.company_name}"
    )

    # Step 2: Chunk
    console.print(f"[bold green]Chunking[/] {file_path.name}...")
    start = time.time()
    chunks = chunk_document(document, chunker_config)
    chunk_time = time.time() - start
    table_chunks = sum(1 for c in chunks if c.metadata.is_table)
    console.print(
        f"  Chunked in {chunk_time:.1f}s — "
        f"{len(chunks)} chunks ({table_chunks} table chunks)"
    )

    # Step 3: Enrich
    console.print(f"[bold yellow]Enriching[/] {file_path.name}...")
    start = time.time()
    enriched = enrich_chunks(chunks, enrichment_config)
    enrich_time = time.time() - start
    console.print(f"  Enriched in {enrich_time:.1f}s — {len(enriched)} chunks")

    return document, enriched


def store_to_databases(
    document: Document,
    enriched_chunks: list[EnrichedChunk],
    ollama_host: str | None = None,
) -> None:
    """Persist document and chunks to PostgreSQL + Qdrant.

    Steps:
    1. Store document metadata + chunk rows in PostgreSQL
    2. Generate embeddings via Ollama
    3. Upsert embeddings + payloads into Qdrant
    4. Record provenance entries
    """
    from app.storage.embeddings import embed_chunks_batched
    from app.storage.postgres import (
        get_sync_session_factory,
        store_chunks,
        store_document,
        store_provenance,
    )
    from app.storage.qdrant import ensure_collection, get_client, upsert_chunks

    console.print(f"[bold magenta]Storing[/] {document.metadata.source_file}...")

    # --- PostgreSQL ---
    start = time.time()
    session_factory = get_sync_session_factory()
    session = session_factory()

    try:
        # Store document
        doc_data = {
            "id": document.id,
            "source_file": document.metadata.source_file,
            "company_name": document.metadata.company_name,
            "filing_type": document.metadata.filing_type,
            "fiscal_year": document.metadata.fiscal_year,
            "file_format": document.metadata.file_format,
            "file_size_bytes": document.metadata.file_size_bytes,
            "total_pages": document.metadata.total_pages,
        }
        store_document(session, doc_data)

        # Store chunks
        chunks_data = [
            {
                "id": c.id,
                "document_id": c.metadata.document_id,
                "text": c.text,
                "summary": c.summary,
                "keywords": c.keywords,
                "keyword_scores": c.keyword_scores,
                "source_file": c.metadata.source_file,
                "company_name": c.metadata.company_name,
                "filing_type": c.metadata.filing_type,
                "fiscal_year": c.metadata.fiscal_year,
                "section_path": c.metadata.section_path,
                "page_number": c.metadata.page_number,
                "chunk_index": c.metadata.chunk_index,
                "is_table": c.metadata.is_table,
            }
            for c in enriched_chunks
        ]
        store_chunks(session, chunks_data)

        # Store provenance
        provenance_entries = [
            {
                "document_id": document.id,
                "chunk_id": None,
                "stage": "parsed",
                "details": {
                    "sections": len(document.sections),
                    "format": document.metadata.file_format,
                },
            },
        ]
        for c in enriched_chunks:
            provenance_entries.append({
                "document_id": document.id,
                "chunk_id": c.id,
                "stage": "enriched",
                "details": {
                    "keywords_count": len(c.keywords),
                    "has_summary": bool(c.summary),
                    "is_table": c.metadata.is_table,
                },
            })
        store_provenance(session, provenance_entries)

        session.commit()
        pg_time = time.time() - start
        console.print(f"  PostgreSQL: stored in {pg_time:.1f}s")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # --- Embeddings ---
    start = time.time()
    texts = [c.text for c in enriched_chunks]
    console.print(f"  Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_chunks_batched(
        texts, batch_size=32, ollama_host=ollama_host
    )
    embed_time = time.time() - start
    console.print(f"  Embeddings generated in {embed_time:.1f}s")

    # --- Qdrant ---
    start = time.time()
    client = get_client()
    ensure_collection(client)

    chunk_ids = [c.id for c in enriched_chunks]
    payloads = [
        {
            "document_id": c.metadata.document_id,
            "source_file": c.metadata.source_file,
            "company_name": c.metadata.company_name,
            "section_path": c.metadata.section_path,
            "page_number": c.metadata.page_number,
            "is_table": c.metadata.is_table,
            "chunk_index": c.metadata.chunk_index,
            "text_preview": c.text[:200],
        }
        for c in enriched_chunks
    ]

    upsert_chunks(client, chunk_ids, embeddings, payloads)
    qdrant_time = time.time() - start
    console.print(f"  Qdrant: upserted in {qdrant_time:.1f}s")


def ingest_directory(
    data_dir: Path,
    chunker_config: ChunkerConfig | None = None,
    enrichment_config: EnrichmentConfig | None = None,
    store: bool = False,
    ollama_host: str | None = None,
    single_file: str | None = None,
) -> dict[str, list[EnrichedChunk]]:
    """Ingest all supported files in a directory.

    Returns:
        Mapping of filename → list of enriched chunks.
    """
    results: dict[str, list[EnrichedChunk]] = {}

    files = sorted(
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in PARSERS
    )

    if single_file:
        files = [f for f in files if f.name == single_file]
        if not files:
            console.print(f"[red]File not found: {single_file}[/]")
            return results

    if not files:
        console.print(f"[red]No supported files found in {data_dir}[/]")
        return results

    console.print(f"\n[bold]Found {len(files)} files to ingest:[/]")
    for f in files:
        console.print(f"  • {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    console.print()

    total_start = time.time()
    for file_path in files:
        try:
            document, enriched = ingest_file(file_path, chunker_config, enrichment_config)
            results[file_path.name] = enriched

            if store:
                store_to_databases(document, enriched, ollama_host=ollama_host)

            console.print()
        except Exception as e:
            console.print(f"[red]Error processing {file_path.name}: {e}[/]")
            logger.exception(f"Failed to ingest {file_path.name}")

    total_time = time.time() - total_start

    # Summary table
    _print_summary(results, total_time)
    return results


def _print_summary(
    results: dict[str, list[EnrichedChunk]], total_time: float
) -> None:
    """Print a summary table of the ingestion results."""
    table = RichTable(title="Ingestion Summary")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Table Chunks", justify="right")
    table.add_column("Avg Keywords", justify="right")
    table.add_column("With Summary", justify="right")

    total_chunks = 0
    for filename, chunks in results.items():
        n = len(chunks)
        total_chunks += n
        table_count = sum(1 for c in chunks if c.metadata.is_table)
        avg_kw = sum(len(c.keywords) for c in chunks) / max(n, 1)
        with_summary = sum(1 for c in chunks if c.summary)

        table.add_row(
            filename,
            str(n),
            str(table_count),
            f"{avg_kw:.1f}",
            str(with_summary),
        )

    console.print(table)
    console.print(
        f"\n[bold green]Done![/] {total_chunks} total chunks "
        f"from {len(results)} files in {total_time:.1f}s"
    )


def _export_chunks(
    results: dict[str, list[EnrichedChunk]], output_path: Path
) -> None:
    """Export enriched chunks to a JSON file."""
    export_data = {}
    for filename, chunks in results.items():
        export_data[filename] = [
            {
                "id": c.id,
                "text": c.text[:500] + "..." if len(c.text) > 500 else c.text,
                "metadata": c.metadata.model_dump(),
                "keywords": c.keywords[:5],
                "summary": c.summary,
            }
            for c in chunks
        ]

    output_path.write_text(json.dumps(export_data, indent=2, default=str))
    console.print(f"[bold]Exported to {output_path}[/]")


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing 10-K filings. Defaults to settings.data_dir.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=1000,
    help="Target chunk size in characters.",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=200,
    help="Overlap between consecutive chunks.",
)
@click.option(
    "--no-summaries/--summaries",
    default=True,
    help="Skip LLM summary generation (default: skip). Use --summaries to enable.",
)
@click.option(
    "--store",
    is_flag=True,
    default=False,
    help="Store chunks in Qdrant + PostgreSQL (requires running services).",
)
@click.option(
    "--ollama-host",
    type=str,
    default=None,
    help="Ollama host URL. Defaults to settings.ollama_host.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Export enriched chunks to JSON file.",
)
@click.option(
    "--file",
    "single_file",
    type=str,
    default=None,
    help="Ingest only this specific file (e.g., 'meta-10k-2024.htm').",
)
def cli(
    data_dir: Path | None,
    chunk_size: int,
    chunk_overlap: int,
    no_summaries: bool,
    store: bool,
    ollama_host: str | None,
    output: Path | None,
    single_file: str | None,
) -> None:
    """Ingest SEC 10-K filings: parse → chunk → enrich [→ store]."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = data_dir or settings.data_dir
    if not data_dir.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/]")
        sys.exit(1)

    chunker_config = ChunkerConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    enrichment_config = EnrichmentConfig(
        ollama_host=ollama_host or settings.ollama_host,
        ollama_model=settings.ollama_model,
        generate_summaries=not no_summaries,
    )

    console.print("[bold]Enterprise Knowledge Copilot — Ingestion Pipeline[/]")
    console.print(f"Data dir: {data_dir}")
    console.print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    console.print(f"Summaries: {'disabled' if no_summaries else 'enabled'}")
    console.print(f"Storage: {'Qdrant + PostgreSQL' if store else 'disabled'}")
    console.print()

    results = ingest_directory(
        data_dir,
        chunker_config,
        enrichment_config,
        store=store,
        ollama_host=ollama_host or settings.ollama_host,
        single_file=single_file,
    )

    if output and results:
        _export_chunks(results, output)


if __name__ == "__main__":
    cli()
