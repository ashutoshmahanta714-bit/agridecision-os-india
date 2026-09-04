"""Fast structural checks before a GitHub commit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "configs/base.yaml",
    "src/agridecision/cli.py",
    "src/agridecision/data/data_gov.py",
    "src/agridecision/features/tabular.py",
    "src/agridecision/models/training.py",
    "src/agridecision/api/app.py",
    "dashboard/app.py",
    ".github/workflows/ci.yml",
]


missing = [name for name in REQUIRED if not (ROOT / name).exists()]
if missing:
    raise SystemExit(f"Missing release files: {missing}")
if (ROOT / ".env").exists():
    raise SystemExit("Unsafe local .env exists in the release directory")

notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
if len(notebooks) < 6:
    raise SystemExit("Expected at least six ordered review notebooks")
for notebook in notebooks:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4:
        raise SystemExit(f"Invalid notebook format: {notebook.name}")
    outputs = [cell.get("outputs", []) for cell in payload["cells"] if cell["cell_type"] == "code"]
    if any(outputs):
        raise SystemExit(f"Notebook contains committed outputs: {notebook.name}")

print(f"Release structure verified: {len(REQUIRED)} required files, {len(notebooks)} notebooks")

