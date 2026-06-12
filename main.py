"""
main.py — RAG application entry point.

Usage:
    python main.py ingest
    python main.py ask "Your question here"
"""

from __future__ import annotations

import sys


def main() -> None:
    from src.utils import get_logger
    log = get_logger(__name__)

    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python main.py ingest")
        print('  python main.py ask "Your question here"')
        return

    command = args[0]

    if command == "ingest":
        from src.ingestion.pipeline import run_ingestion_pipeline
        total = run_ingestion_pipeline()
        log.info(f"Ingestion complete — {total} vectors indexed.")

    elif command == "ask":
        if len(args) < 2:
            print('Usage: python main.py ask "Your question here"')
            return
        question = " ".join(args[1:])
        from src.generation.rag import ask
        response = ask(question, stream=True)

        print(f"\nSources:")
        for src in response.sources:
            print(f"  • {src.source}, page {src.page} (score: {src.score:.4f})")
        print(f"\nLatency: {response.latency_seconds:.2f}s")

    else:
        print(f"Unknown command: '{command}'")


if __name__ == "__main__":
    main()