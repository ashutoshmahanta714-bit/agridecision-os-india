import pandas as pd

from agridecision.data.quality import validate_mandi_data
from agridecision.data.schema import standardise_mandi_frame


def test_schema_aliases_and_quality_quarantine():
    raw = pd.DataFrame(
        [
            {
                "State": "Odisha",
                "District": "Khordha",
                "Market Name": "Bhubaneswar",
                "Commodity": "Onion",
                "Variety": "Other",
                "Date": "01/08/2026",
                "Minimum Price": 1000,
                "Maximum Price": 1500,
                "Modal Price": 1200,
            },
            {
                "State": "Odisha",
                "District": "Khordha",
                "Market Name": "Bhubaneswar",
                "Commodity": "Onion",
                "Variety": "Other",
                "Date": "02/08/2026",
                "Minimum Price": 1400,
                "Maximum Price": 1100,
                "Modal Price": 1200,
            },
        ]
    )
    standard = standardise_mandi_frame(raw)
    accepted, quarantine, report = validate_mandi_data(standard, today=pd.Timestamp("2026-09-01"))
    assert len(accepted) == 1
    assert len(quarantine) == 1
    assert report.inconsistent_price_rows == 1
    assert quarantine.loc[0, "quality_failure"] == "inconsistent_price_order"


def test_mixed_iso_and_day_first_dates_are_parsed_correctly():
    base = {
        "State": "Odisha",
        "District": "Khordha",
        "Market": "Bhubaneswar",
        "Commodity": "Onion",
        "Variety": "Other",
        "Minimum Price": 1000,
        "Maximum Price": 1500,
        "Modal Price": 1200,
    }
    raw = pd.DataFrame([{**base, "Date": "2026-06-01"}, {**base, "Date": "02/06/2026"}])
    parsed = standardise_mandi_frame(raw)
    assert parsed["arrival_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-06-01",
        "2026-06-02",
    ]
