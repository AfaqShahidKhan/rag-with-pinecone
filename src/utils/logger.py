"""
src/utils/logger.py

Structured console logger using `rich`.
Import `get_logger(__name__)` in every module — never use print().
"""

from __future__ import annotations

import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=_console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=True,
        )
    ],
)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)