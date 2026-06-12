from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example → .env and fill in the values."
        )
    return value


def _optional(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str = field(default_factory=lambda: _require("PINECONE_API_KEY"))
    index_name: str = field(default_factory=lambda: _optional("PINECONE_INDEX_NAME", "rag-index"))
    cloud: str = field(default_factory=lambda: _optional("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: _optional("PINECONE_REGION", "us-east-1"))


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = field(default_factory=lambda: _optional("OLLAMA_BASE_URL", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: _optional("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    generation_model: str = field(default_factory=lambda: _optional("OLLAMA_GENERATION_MODEL", "gemma3"))


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int = field(default_factory=lambda: int(_optional("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(_optional("CHUNK_OVERLAP", "64")))


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = field(default_factory=lambda: int(_optional("RETRIEVAL_TOP_K", "5")))


@dataclass(frozen=True)
class Settings:
    pinecone: PineconeSettings = field(default_factory=PineconeSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_raw", self.project_root / "data" / "raw")
        object.__setattr__(self, "data_processed", self.project_root / "data" / "processed")


settings = Settings()