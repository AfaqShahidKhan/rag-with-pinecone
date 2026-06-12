"""
validate_config.py  —  run this after Step 1 to verify your setup.

Usage:
    python validate_config.py
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    console.rule("[bold blue]RAG Config Validation[/bold blue]")

    try:
        from src.config import settings
    except EnvironmentError as exc:
        console.print(f"[bold red]✗ Config error:[/bold red] {exc}")
        raise SystemExit(1)

    from src.utils import get_logger
    log = get_logger(__name__)
    log.info("Logger initialised successfully.")

    table = Table(title="Resolved Settings", show_lines=True)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Key", style="magenta")
    table.add_column("Value", style="green")

    table.add_row("pinecone", "index_name", settings.pinecone.index_name)
    table.add_row("pinecone", "cloud", settings.pinecone.cloud)
    table.add_row("pinecone", "region", settings.pinecone.region)
    table.add_row("pinecone", "api_key", f"{'*' * 8}{settings.pinecone.api_key[-4:]}")
    table.add_row("ollama", "base_url", settings.ollama.base_url)
    table.add_row("ollama", "embed_model", settings.ollama.embed_model)
    table.add_row("ollama", "generation_model", settings.ollama.generation_model)
    table.add_row("chunking", "chunk_size", str(settings.chunking.chunk_size))
    table.add_row("chunking", "chunk_overlap", str(settings.chunking.chunk_overlap))
    table.add_row("retrieval", "top_k", str(settings.retrieval.top_k))
    table.add_row("paths", "data_raw", str(settings.data_raw))
    table.add_row("paths", "data_processed", str(settings.data_processed))

    console.print(table)
    console.print("\n[bold green]✓ Configuration loaded successfully.[/bold green]")


if __name__ == "__main__":
    main()