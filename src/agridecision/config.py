"""Configuration and secret-loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG = Path("configs/base.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the project YAML configuration.

    The path can be supplied explicitly or through ``AGRIDECISION_CONFIG``.
    """

    config_path = Path(path or os.getenv("AGRIDECISION_CONFIG", DEFAULT_CONFIG))
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def get_data_gov_api_key() -> str:
    """Read the Data.gov.in key without printing or persisting it.

    Local runs use ``DATA_GOV_API_KEY``. Google Colab runs can use a secret
    with the same name and notebook access enabled.
    """

    load_dotenv()
    value = os.getenv("DATA_GOV_API_KEY")
    if value:
        return value.strip()

    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get("DATA_GOV_API_KEY")
    except (ImportError, ModuleNotFoundError, KeyError, PermissionError):
        value = None

    if not value:
        raise RuntimeError(
            "DATA_GOV_API_KEY is unavailable. Add it to Colab Secrets or a local .env file."
        )
    return str(value).strip()
