"""
validate_prompt.py — run this after Step 8.

Usage:
    python validate_prompt.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main() -> None:
    console.rule("[bold blue]Prompt Builder Validation[/bold blue]")

    from src.retrieval import search
    from src.generation import build_prompt

    question = "What did Della sell, and why?"

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

    results = search(question, top_k=3)
    package = build_prompt(question, results)

    # Show sources used
    table = Table(title="Sources included in prompt", show_lines=True)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Score", style="yellow", justify="right")
    table.add_column("Source", style="cyan")
    table.add_column("Page", style="magenta", justify="right")

    for i, src in enumerate(package.sources):
        table.add_row(
            str(i + 1),
            f"{src.score:.4f}",
            src.source,
            str(src.page),
        )

    console.print(table)
    console.print(f"\n[dim]Context chunks used: {package.context_chunks_used} | "
                  f"Context chars: {package.context_chars}[/dim]")

    # Show the full assembled prompt
    console.print(Panel(
        package.prompt,
        title="[bold]Assembled RAG Prompt[/bold]",
        border_style="green",
        expand=True,
    ))

    console.print("\n[bold green]✓ Prompt builder working correctly.[/bold green]")


if __name__ == "__main__":
    main()