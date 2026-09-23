"""Small crash-safe helpers for private JSON state files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one state file atomically, preserving the old file on failure."""
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, rendered)


def atomic_write_text(path: Path, contents: str) -> None:
    """Publish complete text atomically; temporary file stays beside target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        # On POSIX this also makes the directory entry durable across a crash.
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
