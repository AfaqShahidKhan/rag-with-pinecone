"""
src/ingestion/loader.py

PDF loading pipeline. Reads PDFs from disk and returns a flat list
of Document objects, one per page, with source metadata attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Document:
    """
    Minimal document unit — a single page of text with provenance metadata.
    Passed through the entire pipeline: load → chunk → embed → upsert → retrieve.
    """
    page_content: str
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path) -> list[Document]:
    """Load a single PDF, returning one Document per page."""
    reader = PdfReader(str(path))
    total = len(reader.pages)
    docs: list[Document] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            log.warning(f"Page {i + 1}/{total} of '{path.name}' yielded no text — skipping.")
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source": path.name,
                "source_path": str(path),
                "page": i + 1,
                "total_pages": total,
            }
        ))

    log.info(f"Loaded '{path.name}': {len(docs)}/{total} pages with text.")
    return docs


def load_pdfs_from_dir(directory: Path) -> list[Document]:
    """Load all PDFs in a directory recursively."""
    pdf_files = sorted(directory.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{directory}'.")

    log.info(f"Found {len(pdf_files)} PDF(s) in '{directory}'.")
    all_docs: list[Document] = []

    for pdf_path in pdf_files:
        all_docs.extend(load_pdf(pdf_path))

    log.info(f"Total pages loaded: {len(all_docs)}")
    return all_docs