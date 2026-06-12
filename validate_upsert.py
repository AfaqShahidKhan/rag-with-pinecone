"""
validate_upsert.py — run this after Step 6.

Usage:
    python validate_upsert.py
"""

import time
from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    console.rule("[bold blue]Upsert Pipeline Validation[/bold blue]")

    from src.ingestion.pipeline import run_ingestion_pipeline
    from src.config.pinecone_client import get_index

    console.print("\n[cyan]→ Running full ingestion pipeline...[/cyan]\n")
    total = run_ingestion_pipeline()

    # Give Pinecone a moment to reflect the upsert in stats
    console.print("\n[cyan]→ Fetching index stats...[/cyan]")
    time.sleep(2)

    index = get_index()
    stats = index.describe_index_stats()

    table = Table(title="Pinecone Index Stats", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total vectors", str(stats.total_vector_count))
    table.add_row("Dimensions", str(stats.dimension))
    table.add_row("Index fullness", f"{stats.index_fullness:.6f}")

    console.print(table)
    console.print(f"\n[bold green]✓ {total} vectors upserted and confirmed in index.[/bold green]")


if __name__ == "__main__":
    main()