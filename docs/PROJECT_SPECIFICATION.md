# Project specification

## Primary user and decision

The first user persona is an agricultural market analyst comparing onion prices across selected Indian mandis. The system supports the question: **what might the price be in seven days, how uncertain is that estimate, and which market has the best risk-adjusted expected net price?**

It does not tell a farmer what they must do and it does not guarantee profit.

## Unit of observation

One canonical row represents one commodity variety, at one mandi, on one calendar date.

## Inputs

Core inputs:

- state, district, market, commodity, variety, grade;
- arrival date;
- minimum, maximum, and modal wholesale prices;
- historical lag and rolling price features.

Optional enrichments:

- reported arrival quantity;
- rainfall, temperature, precipitation, and wind;
- market latitude/longitude and travel distance;
- agricultural bulletin text;
- produce photographs and quality labels;
- preprocessed satellite spectral bands/vegetation indices.

Arrival quantity must not be invented. If the selected source lacks it, the feature remains unavailable until a separately licensed source is joined and documented.

## Targets

Regression target:

`target_modal_price(t) = modal_price(t + 7 calendar days)`

Classification target:

`target_price_shock(t) = 1 if target_modal_price(t) / modal_price(t) - 1 >= 0.15 else 0`

The 15% threshold is provisional. It should be reviewed after real-data EDA and domain consultation, with the original rule retained in the experiment log.

## Acceptance criteria for a professional v1

1. A fixed real-data snapshot has a source URL, retrieval date, checksum, and licence note.
2. Validation reports accepted, rejected, duplicate, missing, impossible, and future-dated rows.
3. Every training feature is available at inference time.
4. Train/calibration/test periods are chronological.
5. The ML forecast is compared with seasonal-naive and recent-median baselines.
6. Uncertainty coverage and performance by market are reported.
7. The API validates missing features and never returns fabricated defaults.
8. The dashboard labels synthetic output and stale data.
9. Unit and integration tests pass in CI.
10. The README reports limitations and does not promise farmer income or model accuracy.

## Expansion gates

- Add tomato/potato only after the onion workflow is reproducible.
- Add 14/30-day horizons only after defining separate metrics and baselines.
- Add a modality only when its source, temporal availability, join key, missingness, and incremental value are measured.
- Deploy recommendations only after costs and risk preferences are user-configurable.

