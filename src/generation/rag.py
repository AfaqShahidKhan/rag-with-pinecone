"""
src/generation/rag.py

Top-level RAG query interface.
Call ask() from main.py, CLI, or Streamlit UI.
"""

from __future__ import annotations

from src.generation.generator import RAGResponse, generate
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import search
from src.utils.logger import get_logger

log = get_logger(__name__)


def ask(question: str, top_k: int | None = None, stream: bool = True) -> RAGResponse:
    """
    Full RAG pipeline: search → prompt → generate.

    Args:
        question: Natural language question.
        top_k:    Number of context chunks to retrieve.
        stream:   Stream tokens to stdout while generating.

    Returns:
        RAGResponse with answer and source metadata.
    """
    log.info(f"RAG query: '{question}'")

    results = search(question, top_k=top_k)
    package = build_prompt(question, results)
    response = generate(package, stream=stream)

    return response