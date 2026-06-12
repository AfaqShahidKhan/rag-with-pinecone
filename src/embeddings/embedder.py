"""
src/embeddings/embedder.py

Generates embeddings for document chunks using Ollama.
Returns (Document, vector) pairs ready for Pinecone upsert.
"""

from __future__ import annotations

import time
from typing import Generator

import ollama
from tqdm import tqdm

from src.config.settings import settings
from src.ingestion.loader import Document
from src.utils.logger import get_logger

log = get_logger(__name__)

# How many chunks to send to Ollama in one batch
DEFAULT_BATCH_SIZE = 8


def _embed_batch(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Embed a batch of texts, with simple retry logic."""
    for attempt in range(retries):
        try:
            response = ollama.embed(
                model=settings.ollama.embed_model,
                input=texts,
            )
            return response.embeddings
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                log.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"All {retries} embedding attempts failed for batch.")
                raise


def embed_documents(
    chunks: list[Document],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[tuple[Document, list[float]]]:
    """
    Embed all chunks in batches.
    Returns a list of (Document, embedding_vector) pairs.
    """
    results: list[tuple[Document, list[float]]] = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    log.info(
        f"Embedding {len(chunks)} chunks in {total_batches} batches "
        f"using '{settings.ollama.embed_model}'..."
    )

    with tqdm(total=len(chunks), desc="Embedding", unit="chunk") as progress:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [doc.page_content for doc in batch]

            vectors = _embed_batch(texts)

            for doc, vector in zip(batch, vectors):
                results.append((doc, vector))

            progress.update(len(batch))

    log.info(f"Embedding complete. {len(results)} vectors generated (dim={len(results[0][1])}).")
    return results