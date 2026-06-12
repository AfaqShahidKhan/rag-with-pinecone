"""
src/config/pinecone_client.py

Pinecone client factory and serverless index bootstrap.
Call get_index() to get a live IndexModel handle for upsert/query operations.
"""

from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# nomic-embed-text produces 768-dimensional vectors — fixed at design time
EMBEDDING_DIMENSION = 768
METRIC = "cosine"


def get_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone.api_key)


def get_index():
    """
    Returns a live Pinecone index handle.
    Creates the serverless index if it doesn't already exist.
    """
    pc = get_client()
    index_name = settings.pinecone.index_name
    existing = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        log.info(f"Index '{index_name}' not found — creating serverless index...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=settings.pinecone.cloud,
                region=settings.pinecone.region,
            ),
        )
        log.info(f"Index '{index_name}' created successfully.")
    else:
        log.info(f"Index '{index_name}' already exists — skipping creation.")

    return pc.Index(index_name)