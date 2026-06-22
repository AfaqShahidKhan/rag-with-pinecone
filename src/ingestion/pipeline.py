"""
src/ingestion/pipeline.py

Top-level ingestion orchestrator: load → chunk → embed → upsert.
Call run_ingestion_pipeline() from main.py or CLI.
"""

from __future__ import annotations

from pathlib import Path

from src.config.settings import settings
from src.embeddings.embedder import embed_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_pdf, load_docx, load_documents_from_dir
from src.ingestion.upsert import upsert_embeddings
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_ingestion_pipeline(source: Path | None = None) -> int:
    """
    Run the full ingestion pipeline.

    Args:
        source: Path to a single PDF/DOCX file, or a directory.
                Defaults to settings.data_raw if not provided.

    Returns:
        Number of vectors upserted.
    """
    target = source or settings.data_raw

    if target.is_file():
        suffix = target.suffix.lower()
        if suffix == ".pdf":
            docs = load_pdf(target)
        elif suffix == ".docx":
            docs = load_docx(target)
        else:
            raise ValueError(f"Unsupported file type: '{suffix}'. Supported: .pdf, .docx")
    else:
        docs = load_documents_from_dir(target)  # ← handles both .pdf and .docx

    chunks = chunk_documents(docs)
    embedded = embed_documents(chunks)
    total = upsert_embeddings(embedded)

    log.info(f"Pipeline complete — {total} vectors indexed from '{target.name}'.")
    return total

    