"""
src/retrieval/retriever.py

Semantic search over the Pinecone index.
Embeds a query, fetches top-K similar chunks, returns typed results.
"""

from __future__ import annotations

from dataclasses import dataclass

import ollama

from src.config.pinecone_client import get_index
from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    page: int
    chunk_index: int


def _embed_query(query: str) -> list[float]:
    response = ollama.embed(
        model=settings.ollama.embed_model,
        input=[query],
    )
    return response.embeddings[0]


def search(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """
    Semantic search over the Pinecone index.

    Args:
        query:           Natural language query string.
        top_k:           Number of results to return. Defaults to settings.retrieval.top_k.
        score_threshold: Optional minimum cosine similarity score (0-1).
                         Results below this are filtered out.

    Returns:
        List of SearchResult ordered by descending similarity score.
    """
    k = top_k or settings.retrieval.top_k
    query_vector = _embed_query(query)

    index = get_index()
    response = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
    )

    results: list[SearchResult] = []
    for match in response.matches:
        score = match.score
        if score_threshold is not None and score < score_threshold:
            log.debug(f"Skipping match (score={score:.4f} < threshold={score_threshold})")
            continue

        meta = match.metadata or {}
        results.append(SearchResult(
            text=meta.get("text", ""),
            score=score,
            source=meta.get("source", "unknown"),
            page=int(meta.get("page", 0)),
            chunk_index=int(meta.get("chunk_index", 0)),
        ))

    log.info(f"Query returned {len(results)} results (top_k={k}).")
    return results