"""
validate_generation.py — run this after Step 9.

Usage:
    python validate_generation.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

TEST_QUESTIONS = [
    "What did Della sell to buy Jim's gift?",
    "What gift did Jim buy for Della?",
    "How much money did Della have at the start of the story?",
]


def main() -> None:
    console.rule("[bold blue]RAG Generation Validation[/bold blue]")

    from src.generation.rag import ask

    for question in TEST_QUESTIONS:
        console.rule(f"[cyan]{question}[/cyan]", style="cyan")
        console.print()

        response = ask(question, stream=True)

        # Sources table
        table = Table(title="Sources", show_lines=True, width=80)
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Score", style="yellow", justify="right", width=8)
        table.add_column("Source", style="cyan")
        table.add_column("Page", style="magenta", justify="right", width=6)

        for i, src in enumerate(response.sources):
            table.add_row(
                str(i + 1),
                f"{src.score:.4f}",
                src.source,
                str(src.page),
            )

        console.print(table)
        console.print(
            f"[dim]Model: {response.model} | "
            f"Latency: {response.latency_seconds:.2f}s | "
            f"Chunks: {response.context_chunks_used}[/dim]\n"
        )

    console.print("[bold green]✓ RAG generation complete.[/bold green]")


if __name__ == "__main__":
    main()