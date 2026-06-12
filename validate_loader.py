"""
validate_loader.py — run this after Step 3.

Usage:
    python validate_loader.py
"""

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    console.rule("[bold blue]PDF Loader Validation[/bold blue]")

    from src.config import settings
    from src.ingestion import load_pdfs_from_dir

    docs = load_pdfs_from_dir(settings.data_raw)

    table = Table(title=f"Loaded Documents ({len(docs)} pages)", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Page", style="magenta", justify="right")
    table.add_column("Total Pages", style="yellow", justify="right")
    table.add_column("Chars", style="green", justify="right")
    table.add_column("Preview", style="white", max_width=60)

    for doc in docs:
        table.add_row(
            doc.metadata["source"],
            str(doc.metadata["page"]),
            str(doc.metadata["total_pages"]),
            str(len(doc.page_content)),
            doc.page_content[:80].replace("\n", " "),
        )

    console.print(table)
    console.print(f"\n[bold green]✓ Loaded {len(docs)} pages successfully.[/bold green]")


if __name__ == "__main__":
    main()