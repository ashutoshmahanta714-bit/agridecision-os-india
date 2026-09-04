"""Immutable source-snapshot metadata and checksums."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_snapshot_manifest(
    data_path: str | Path,
    *,
    source_url: str,
    source_name: str,
    license_name: str = "Government Open Data License - India",
    is_synthetic: bool = False,
) -> dict:
    path = Path(data_path)
    return {
        "source_name": source_name,
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "license": license_name,
        "is_synthetic": is_synthetic,
    }


def write_snapshot_manifest(manifest: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
