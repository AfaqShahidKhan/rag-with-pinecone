"""
src/generation/prompt_builder.py

Assembles retrieved chunks into a structured RAG prompt.
Returns a PromptPackage containing the final prompt and source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.retriever import SearchResult
from src.utils.logger import get_logger

log = get_logger(__name__)

# Approximate character budget for context (fits comfortably in Gemma3's context window)
MAX_CONTEXT_CHARS = 6000

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not present in the context, say "I don't have enough information to answer that."
Do not make up facts. Be concise and direct."""

PROMPT_TEMPLATE = """{system}

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class PromptPackage:
    prompt: str
    question: str
    sources: list[SearchResult]
    context_chunks_used: int
    context_chars: int


def build_prompt(question: str, results: list[SearchResult]) -> PromptPackage:
    """
    Construct a RAG prompt from a question and retrieved search results.

    Args:
        question: The user's question.
        results:  Ranked search results from the retriever.

    Returns:
        PromptPackage with the assembled prompt and metadata.
    """
    if not results:
        log.warning("No search results provided — prompt will have empty context.")

    context_parts: list[str] = []
    total_chars = 0
    used = 0

    for i, result in enumerate(results):
        chunk_text = (
            f"[{i + 1}] (Source: {result.source}, Page {result.page})\n"
            f"{result.text}"
        )
        chunk_chars = len(chunk_text)

        if total_chars + chunk_chars > MAX_CONTEXT_CHARS:
            log.warning(
                f"Context budget reached at chunk {i + 1}/{len(results)} "
                f"({total_chars} chars) — truncating."
            )
            break

        context_parts.append(chunk_text)
        total_chars += chunk_chars
        used += 1

    context = "\n\n".join(context_parts) if context_parts else "No context available."

    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )

    log.info(
        f"Prompt built — {used}/{len(results)} chunks used, "
        f"{total_chars} context chars."
    )

    return PromptPackage(
        prompt=prompt,
        question=question,
        sources=results[:used],
        context_chunks_used=used,
        context_chars=total_chars,
    )