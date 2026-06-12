"""
validate_retrieval.py — run this after Step 7.

Usage:
    python validate_retrieval.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

TEST_QUERIES = [
    "How much money did Della have?",
    "What did Jim give Della as a gift?",
    "What did Della sell to buy Jim's gift?",
]


def main() -> None:
    console.rule("[bold blue]Semantic Search Validation[/bold blue]")

    from src.retrieval import search

    for query in TEST_QUERIES:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")

        results = search(query, top_k=3)

        table = Table(show_lines=True, expand=True)
        table.add_column("Rank", style="dim", justify="right", width=4)
        table.add_column("Score", style="yellow", justify="right", width=7)
        table.add_column("Source", style="cyan", width=26)
        table.add_column("Page", style="magenta", justify="right", width=4)
        table.add_column("Text Preview", style="white")

        for i, result in enumerate(results):
            table.add_row(
                str(i + 1),
                f"{result.score:.4f}",
                result.source,
                str(result.page),
                result.text[:120].replace("\n", " "),
            )

        console.print(table)

    # Show one full result as a sanity check
    console.print("\n[bold cyan]Full top result for last query:[/bold cyan]")
    results = search(TEST_QUERIES[-1], top_k=1)
    if results:
        console.print(Panel(
            results[0].text,
            title=f"[bold]Score: {results[0].score:.4f} | {results[0].source} p.{results[0].page}[/bold]",
            border_style="green",
        ))

    console.print("\n[bold green]✓ Semantic search working correctly.[/bold green]")


if __name__ == "__main__":
    main()