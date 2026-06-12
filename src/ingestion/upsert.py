"""
src/ingestion/upsert.py

Upserts embedded document chunks into Pinecone.
Vector IDs are deterministic — re-ingesting the same PDF is idempotent.
"""

from __future__ import annotations

import hashlib

from tqdm import tqdm

from src.config.pinecone_client import get_index
from src.ingestion.loader import Document
from src.utils.logger import get_logger

log = get_logger(__name__)

UPSERT_BATCH_SIZE = 100


def _make_vector_id(doc: Document) -> str:
    """Stable ID derived from source + page + chunk position."""
    key = f"{doc.metadata['source']}::p{doc.metadata['page']}::c{doc.metadata['chunk_index']}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def upsert_embeddings(embedded: list[tuple[Document, list[float]]]) -> int:
    """
    Upsert (Document, vector) pairs into Pinecone.
    Returns the total number of vectors upserted.
    """
    index = get_index()
    vectors = []

    for doc, vector in embedded:
        vectors.append({
            "id": _make_vector_id(doc),
            "values": vector,
            "metadata": {
                "source": doc.metadata["source"],
                "page": doc.metadata["page"],
                "total_pages": doc.metadata["total_pages"],
                "chunk_index": doc.metadata["chunk_index"],
                "chunk_total": doc.metadata["chunk_total"],
                "text": doc.page_content,
            },
        })

    total = len(vectors)
    log.info(f"Upserting {total} vectors to Pinecone in batches of {UPSERT_BATCH_SIZE}...")

    with tqdm(total=total, desc="Upserting", unit="vec") as progress:
        for i in range(0, total, UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            index.upsert(vectors=batch)
            progress.update(len(batch))

    log.info(f"Upsert complete — {total} vectors in index.")
    return total