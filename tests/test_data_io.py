import json

import pandas as pd

from agridecision.data.bulletins import load_bulletin_csv
from agridecision.data.csv_loader import load_mandi_file
from agridecision.data.data_gov import DataGovClient
from agridecision.data.provenance import create_snapshot_manifest, sha256_file
from agridecision.data.readiness import assess_training_readiness
from agridecision.data.sqlite_store import query_frame, write_mandi_observations
from agridecision.data.weather import merge_market_weather


def _rows():
    return [
        {
            "state": "Odisha",
            "district": "Khordha",
            "market": "Bhubaneswar",
            "commodity": "Onion",
            "variety": "Other",
            "arrival_date": "01/08/2026",
            "min_price": 1000,
            "max_price": 1400,
            "modal_price": 1200,
        },
        {
            "state": "Odisha",
            "district": "Khordha",
            "market": "Bhubaneswar",
            "commodity": "Onion",
            "variety": "Other",
            "arrival_date": "02/08/2026",
            "min_price": 1100,
            "max_price": 1500,
            "modal_price": 1300,
        },
    ]


def test_csv_loader_sqlite_and_bulletins(tmp_path):
    source = tmp_path / "prices.csv"
    pd.DataFrame(_rows()).to_csv(source, index=False)
    loaded = load_mandi_file(source)
    assert str(loaded["arrival_date"].dtype).startswith("datetime64")
    database = tmp_path / "prices.db"
    write_mandi_observations(loaded, database)
    assert query_frame(database, "SELECT COUNT(*) AS n FROM mandi_prices").loc[0, "n"] == 2

    bulletin_path = tmp_path / "bulletins.csv"
    pd.DataFrame({"published_date": ["2026-08-01"], "text": [" Heavy rainfall "]}).to_csv(
        bulletin_path, index=False
    )
    bulletin = load_bulletin_csv(bulletin_path)
    assert bulletin.loc[0, "text"] == "Heavy rainfall"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.params = []

    def get(self, url, params, timeout):
        self.params.append(params)
        return FakeResponse(next(self.payloads))


def test_data_gov_pagination_and_filter_translation():
    client = DataGovClient("secret", "resource", page_size=2, request_delay_seconds=0)
    fake = FakeSession(
        [
            {"total": 3, "records": _rows()},
            {"total": 3, "records": [_rows()[0]]},
        ]
    )
    client.session = fake
    pages = list(client.iter_pages(filters={"commodity": "Onion"}))
    assert [len(page) for page in pages] == [2, 1]
    assert fake.params[0]["filters[commodity]"] == "Onion"
    assert fake.params[1]["offset"] == 2


def test_weather_join_and_readiness():
    mandi = pd.DataFrame(_rows())
    mandi["arrival_date"] = pd.to_datetime(mandi["arrival_date"], dayfirst=True)
    weather = pd.DataFrame(
        {"arrival_date": pd.to_datetime(["2026-08-01", "2026-08-02"]), "rainfall_mm": [1, 2]}
    )
    joined = merge_market_weather(mandi, weather)
    assert joined["rainfall_mm"].tolist() == [1, 2]
    _, report = assess_training_readiness(
        mandi, minimum_dates_per_market=2, minimum_eligible_markets=1
    )
    assert report.ready_for_forecasting
    assert json.loads(json.dumps(report.to_dict()))["unique_dates"] == 2


def test_snapshot_manifest_contains_checksum(tmp_path):
    source = tmp_path / "snapshot.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = create_snapshot_manifest(
        source, source_url="https://example.org/data", source_name="Example", is_synthetic=True
    )
    assert manifest["sha256"] == sha256_file(source)
    assert manifest["size_bytes"] == source.stat().st_size
    assert manifest["is_synthetic"] is True
