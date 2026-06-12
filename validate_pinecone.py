"""
validate_pinecone.py — run this after Step 2 to verify Pinecone connectivity.

Usage:
    python validate_pinecone.py
"""

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    console.rule("[bold blue]Pinecone Connection Validation[/bold blue]")

    from src.config.pinecone_client import get_client, get_index

    console.print("\n[cyan]→ Connecting to Pinecone...[/cyan]")
    get_index()

    pc = get_client()
    indexes = pc.list_indexes()

    table = Table(title="Pinecone Indexes", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Dimension", style="magenta")
    table.add_column("Metric", style="green")
    table.add_column("Status", style="yellow")

    for idx in indexes:
        table.add_row(
            idx.name,
            str(idx.dimension),
            idx.metric,
            idx.status.get("state", "unknown") if isinstance(idx.status, dict) else str(idx.status),
        )

    console.print(table)
    console.print("\n[bold green]✓ Pinecone connection successful.[/bold green]")


if __name__ == "__main__":
    main()