"""Small IO helpers: YAML config, JSONL results, timestamps."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import yaml


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(path: str | os.PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class JsonlWriter:
    """Append-mode JSONL writer so long runs are crash-safe (rows flushed as produced)."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._f = open(self.path, "w", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
