"""
validate_embedder.py — run this after Step 5.

Usage:
    python validate_embedder.py
"""

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    console.rule("[bold blue]Embedder Validation[/bold blue]")

    from src.config import settings
    from src.ingestion import load_pdfs_from_dir, chunk_documents
    from src.embeddings import embed_documents

    docs = load_pdfs_from_dir(settings.data_raw)
    chunks = chunk_documents(docs)

    # Only embed first 5 chunks to keep validation fast
    sample = chunks[:5]
    console.print(f"\n[cyan]→ Embedding {len(sample)} sample chunks...[/cyan]\n")

    embedded = embed_documents(sample)

    table = Table(title="Embedding Results (sample)", show_lines=True)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Source", style="cyan")
    table.add_column("Page", style="magenta", justify="right")
    table.add_column("Chunk", style="yellow", justify="right")
    table.add_column("Vector Dim", style="green", justify="right")
    table.add_column("Vector Preview", style="white")

    for i, (doc, vector) in enumerate(embedded):
        table.add_row(
            str(i),
            doc.metadata["source"],
            str(doc.metadata["page"]),
            str(doc.metadata["chunk_index"]),
            str(len(vector)),
            f"[{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f} ...]",
        )

    console.print(table)
    console.print(f"\n[bold green]✓ Embeddings verified — dimension={len(embedded[0][1])}.[/bold green]")


if __name__ == "__main__":
    main()