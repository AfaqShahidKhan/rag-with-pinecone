"""
validate_eval.py — run this after Step 10.

Usage:
    python validate_eval.py            # run full eval suite
    python validate_eval.py debug      # debug a single query
"""

import sys
from rich.console import Console

console = Console()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"

    if mode == "debug":
        from src.utils import debug_query
        debug_query("What did Jim sell to buy Della's gift?", top_k=5)

    else:
        from src.utils import run_eval
        results = run_eval(stream=False)

        failed = [r for r in results if not r.passed]
        if failed:
            console.print("\n[bold yellow]Failed cases to investigate:[/bold yellow]")
            for r in failed:
                console.print(f"  • {r.case.question}")
                console.print(f"    Expected one of: {r.case.expected_keywords}")
                console.print(f"    Got: {r.answer}")


if __name__ == "__main__":
    main()