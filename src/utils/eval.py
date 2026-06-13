"""
src/utils/eval.py

Evaluation and debugging utilities for the RAG pipeline.
Provides retrieval inspection, batch eval, and index health checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.generation.rag import ask
from src.retrieval.retriever import SearchResult, search
from src.utils.logger import get_logger

log = get_logger(__name__)
console = Console()


# ── Test case definition ──────────────────────────────────────────────────────

@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]  # answer must contain at least one
    notes: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    passed: bool
    matched_keywords: list[str]
    latency_seconds: float
    top_score: float
    sources: list[SearchResult] = field(default_factory=list)


# ── Default test suite ────────────────────────────────────────────────────────

DEFAULT_EVAL_SUITE: list[EvalCase] = [
    EvalCase(
        question="How much money did Della have at the start?",
        expected_keywords=["1.87", "one dollar", "eighty-seven"],
        notes="Opening fact — should be high confidence",
    ),
    EvalCase(
        question="What did Della sell to get money?",
        expected_keywords=["hair", "brown hair"],
        notes="Core plot point",
    ),
    EvalCase(
        question="What gift did Jim buy for Della?",
        expected_keywords=["comb", "combs", "hair comb", "watch chain", "chain"],
        notes="Ironic gift reveal — context may surface watch chain instead of combs",
    ),
    EvalCase(
        question="What did Jim sell to buy Della's gift?",
        expected_keywords=["watch", "gold watch"],
        notes="Jim's sacrifice — tests retrieval of Jim-focused chunks",
    ),
    EvalCase(
        question="Who cut Della's hair?",
        expected_keywords=["sofronie", "mrs. sofronie"],
        notes="Specific named character",
    ),
    EvalCase(
        question="What is the theme of the story?",
        expected_keywords=["love", "sacrifice", "gift", "wise", "magi"],
        notes="Abstract question — tests generalization",
    ),
]


# ── Retrieval debugger ────────────────────────────────────────────────────────

def debug_query(question: str, top_k: int = 5) -> None:
    """
    Print a full diagnostic trace for a single query:
    retrieved chunks, scores, and the assembled prompt.
    """
    from src.generation.prompt_builder import build_prompt

    console.rule(f"[bold cyan]Debug: {question}[/bold cyan]")

    results = search(question, top_k=top_k)

    table = Table(title=f"Retrieved Chunks (top {top_k})", show_lines=True, expand=True)
    table.add_column("Rank", style="dim", justify="right", width=4)
    table.add_column("Score", style="yellow", justify="right", width=7)
    table.add_column("Page", style="magenta", justify="right", width=4)
    table.add_column("Chunk", style="cyan", justify="right", width=5)
    table.add_column("Text", style="white")

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            f"{r.score:.4f}",
            str(r.page),
            str(r.chunk_index),
            r.text[:200].replace("\n", " "),
        )

    console.print(table)

    package = build_prompt(question, results)
    console.print(Panel(
        package.prompt,
        title="[bold]Full Prompt[/bold]",
        border_style="dim",
        expand=True,
    ))


# ── Batch evaluator ───────────────────────────────────────────────────────────

def run_eval(
    suite: list[EvalCase] | None = None,
    stream: bool = False,
) -> list[EvalResult]:
    """
    Run a batch evaluation suite and print a results table.

    Args:
        suite:  List of EvalCase. Defaults to DEFAULT_EVAL_SUITE.
        stream: Whether to stream generation output (noisy in batch mode).

    Returns:
        List of EvalResult with pass/fail per case.
    """
    cases = suite or DEFAULT_EVAL_SUITE
    results: list[EvalResult] = []

    console.rule("[bold blue]RAG Evaluation Suite[/bold blue]")
    console.print(f"Running {len(cases)} test cases...\n")

    for i, case in enumerate(cases):
        console.print(f"[cyan][{i+1}/{len(cases)}][/cyan] {case.question}")

        start = time.perf_counter()
        response = ask(case.question, stream=stream)
        latency = time.perf_counter() - start

        answer_lower = response.answer.lower()
        matched = [kw for kw in case.expected_keywords if kw.lower() in answer_lower]
        passed = len(matched) > 0
        top_score = response.sources[0].score if response.sources else 0.0

        result = EvalResult(
            case=case,
            answer=response.answer,
            passed=passed,
            matched_keywords=matched,
            latency_seconds=latency,
            top_score=top_score,
            sources=response.sources,
        )
        results.append(result)

        status = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        console.print(f"  {status} — Answer: [italic]{response.answer[:80]}[/italic]")
        if matched:
            console.print(f"  Matched: {matched}")
        console.print()

    _print_eval_summary(results)
    return results


def _print_eval_summary(results: list[EvalResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_latency = sum(r.latency_seconds for r in results) / total if total else 0
    avg_score = sum(r.top_score for r in results) / total if total else 0

    table = Table(title="Evaluation Summary", show_lines=True)
    table.add_column("Question", style="cyan", max_width=45)
    table.add_column("Result", justify="center", width=6)
    table.add_column("Answer", style="white", max_width=35)
    table.add_column("Top Score", style="yellow", justify="right", width=9)
    table.add_column("Latency", style="dim", justify="right", width=8)

    for r in results:
        status = "✓" if r.passed else "✗"
        style = "green" if r.passed else "red"
        table.add_row(
            r.case.question,
            f"[{style}]{status}[/{style}]",
            r.answer[:60],
            f"{r.top_score:.4f}",
            f"{r.latency_seconds:.1f}s",
        )

    console.print(table)
    console.print(
        f"\n[bold]Score: {passed}/{total} "
        f"({'[green]' if passed == total else '[yellow]'}{passed/total*100:.0f}%"
        f"{'[/green]' if passed == total else '[/yellow]'})[/bold] | "
        f"Avg latency: {avg_latency:.1f}s | "
        f"Avg top retrieval score: {avg_score:.4f}"
    )