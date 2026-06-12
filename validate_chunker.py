"""
validate_chunker.py — run this after Step 4.

Usage:
    python validate_chunker.py
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def main() -> None:
    console.rule("[bold blue]Chunker Validation[/bold blue]")

    from src.config import settings
    from src.ingestion import load_pdfs_from_dir, chunk_documents

    docs = load_pdfs_from_dir(settings.data_raw)
    chunks = chunk_documents(docs)

    # Summary table
    table = Table(title=f"Chunks ({len(chunks)} total)", show_lines=True)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Source", style="cyan")
    table.add_column("Page", style="magenta", justify="right")
    table.add_column("Chunk", style="yellow", justify="right")
    table.add_column("Chars", style="green", justify="right")
    table.add_column("Preview", style="white", max_width=60)

    for i, chunk in enumerate(chunks):
        table.add_row(
            str(i),
            chunk.metadata["source"],
            str(chunk.metadata["page"]),
            f"{chunk.metadata['chunk_index'] + 1}/{chunk.metadata['chunk_total']}",
            str(len(chunk.page_content)),
            chunk.page_content[:80].replace("\n", " "),
        )

    console.print(table)

    # Show one full chunk as a sanity check
    console.print(Panel(
        chunks[1].page_content,
        title="[bold]Sample Chunk (index 1)[/bold]",
        border_style="green",
    ))

    console.print(f"\n[bold green]✓ {len(chunks)} chunks ready for embedding.[/bold green]")


if __name__ == "__main__":
    main()