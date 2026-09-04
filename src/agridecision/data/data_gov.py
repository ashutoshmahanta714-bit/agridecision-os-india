"""Fault-tolerant Data.gov.in client with pagination and resumable downloads."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agridecision.data.schema import standardise_mandi_frame

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int | None], None]


class DataGovClient:
    """Retrieve a resource without exposing the API key in logs."""

    def __init__(
        self,
        api_key: str,
        resource_id: str,
        *,
        base_url: str = "https://api.data.gov.in/resource",
        page_size: int = 500,
        request_delay_seconds: float = 1.5,
        timeout: tuple[float, float] = (10.0, 90.0),
        max_retries: int = 5,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")

        self._api_key = api_key
        self.resource_id = resource_id
        self.url = f"{base_url.rstrip('/')}/{resource_id}"
        self.page_size = page_size
        self.request_delay_seconds = request_delay_seconds
        self.timeout = timeout

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=2.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": "AgriDecision-OS/0.1"})

    def fetch_page(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> dict:
        params: dict[str, str | int] = {
            "api-key": self._api_key,
            "format": "json",
            "offset": offset,
            "limit": limit or self.page_size,
        }
        for name, value in (filters or {}).items():
            params[f"filters[{name}]"] = value

        response = self.session.get(self.url, params=params, timeout=self.timeout)
        response.raise_for_status()
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise RuntimeError("Data.gov.in returned a non-JSON response") from exc
        if "records" not in payload or not isinstance(payload["records"], list):
            raise RuntimeError("Data.gov.in response does not contain a records list")
        return payload

    def iter_pages(
        self,
        *,
        filters: dict[str, str] | None = None,
        start_offset: int = 0,
        max_pages: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Iterator[pd.DataFrame]:
        offset = start_offset
        page_number = 0
        total: int | None = None

        while max_pages is None or page_number < max_pages:
            payload = self.fetch_page(offset=offset, filters=filters)
            records = payload["records"]
            if total is None and payload.get("total") is not None:
                try:
                    total = int(payload["total"])
                except (TypeError, ValueError):
                    total = None
            if not records:
                break

            frame = pd.DataFrame.from_records(records)
            yield frame
            offset += len(frame)
            page_number += 1
            if progress:
                progress(offset, total)
            if len(frame) < self.page_size or (total is not None and offset >= total):
                break
            time.sleep(self.request_delay_seconds)

    def download_csv(
        self,
        output_path: str | Path,
        *,
        filters: dict[str, str] | None = None,
        max_pages: int | None = None,
        resume: bool = True,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        """Download pages to a checkpointed partial file, then canonicalise."""

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        checkpoint = destination.with_suffix(destination.suffix + ".checkpoint.json")

        start_offset = 0
        if resume and partial.exists() and checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            start_offset = int(state.get("next_offset", 0))
            LOGGER.info("Resuming mandi download at offset %s", start_offset)
        elif not resume:
            partial.unlink(missing_ok=True)
            checkpoint.unlink(missing_ok=True)

        wrote_header = partial.exists() and partial.stat().st_size > 0
        current_offset = start_offset
        for page in self.iter_pages(
            filters=filters,
            start_offset=start_offset,
            max_pages=max_pages,
            progress=progress,
        ):
            page.to_csv(partial, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            current_offset += len(page)
            checkpoint.write_text(
                json.dumps({"next_offset": current_offset, "filters": filters or {}}, indent=2),
                encoding="utf-8",
            )

        if not partial.exists():
            raise RuntimeError("The API returned no records; no output file was created")

        result = standardise_mandi_frame(pd.read_csv(partial, low_memory=False))
        result.to_csv(destination, index=False)
        partial.unlink(missing_ok=True)
        checkpoint.unlink(missing_ok=True)
        return result
