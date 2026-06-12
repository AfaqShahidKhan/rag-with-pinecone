"""
src/ingestion/chunker.py

Splits raw page Documents into overlapping chunks ready for embedding.
Also handles text normalization to clean up common PDF extraction artifacts.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.ingestion.loader import Document
from src.utils.logger import get_logger

log = get_logger(__name__)


def _clean_text(text: str) -> str:
    """
    Normalize common PDF extraction artifacts:
    - Collapse spaced-out characters: 'T h e' → 'The'
    - Normalize whitespace and line breaks
    """
    # Fix spaced characters: single chars separated by single spaces
    # e.g. "T h e  G i f t" → "The Gift"
    text = re.sub(r"(?<!\w)((\w) )+(\w)(?!\w)", lambda m: m.group(0).replace(" ", ""), text)

    # Normalize multiple newlines to double (paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Takes a list of page-level Documents and returns a flat list
    of chunk-level Documents with updated metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []

    for doc in docs:
        cleaned = _clean_text(doc.page_content)
        if not cleaned:
            continue

        splits = splitter.split_text(cleaned)

        for i, chunk_text in enumerate(splits):
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_total": len(splits),
                }
            ))

    log.info(
        f"Chunked {len(docs)} pages → {len(all_chunks)} chunks "
        f"(size={settings.chunking.chunk_size}, overlap={settings.chunking.chunk_overlap})"
    )
    return all_chunks