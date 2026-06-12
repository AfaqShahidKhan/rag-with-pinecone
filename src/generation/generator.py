"""
src/generation/generator.py

Generates answers from RAG prompts using Ollama.
Streams output to console for responsive UX with local models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import ollama

from src.config.settings import settings
from src.generation.prompt_builder import PromptPackage
from src.retrieval.retriever import SearchResult
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    question: str
    sources: list[SearchResult]
    model: str
    latency_seconds: float
    context_chunks_used: int


def generate(package: PromptPackage, stream: bool = True) -> RAGResponse:
    """
    Generate an answer from a PromptPackage using Ollama.

    Args:
        package: Assembled prompt from prompt_builder.build_prompt().
        stream:  If True, streams tokens to stdout as they arrive.

    Returns:
        RAGResponse with the complete answer and metadata.
    """
    model = settings.ollama.generation_model
    log.info(f"Generating answer with '{model}' (stream={stream})...")

    start = time.perf_counter()
    answer_parts: list[str] = []

    if stream:
        print()  # newline before streamed output
        for chunk in ollama.generate(model=model, prompt=package.prompt, stream=True):
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            answer_parts.append(token)
        print("\n")  # newline after streamed output
    else:
        response = ollama.generate(model=model, prompt=package.prompt, stream=False)
        answer_parts.append(response.get("response", ""))

    latency = time.perf_counter() - start
    answer = "".join(answer_parts).strip()

    log.info(f"Generation complete in {latency:.2f}s — {len(answer)} chars.")

    return RAGResponse(
        answer=answer,
        question=package.question,
        sources=package.sources,
        model=model,
        latency_seconds=latency,
        context_chunks_used=package.context_chunks_used,
    )