# Data dictionary

| Column | Type | Required | Meaning | Validation |
|---|---|---:|---|---|
| `state` | string | Yes | State/UT reported by source | Non-empty |
| `district` | string | Yes | District reported by source | Non-empty |
| `market` | string | Yes | Mandi/market name | Non-empty |
| `commodity` | string | Yes | Commodity, initially Onion | Non-empty |
| `variety` | string | Yes | Reported commodity variety | Non-empty |
| `grade` | string | No | Reported grade | May be missing |
| `arrival_date` | date | Yes | Observation date | Parseable and not future-dated at ingestion |
| `min_price` | float | Yes | Minimum wholesale price | Positive, ≤ modal price |
| `max_price` | float | Yes | Maximum wholesale price | Positive, ≥ modal price |
| `modal_price` | float | Yes | Most representative reported price | Positive, between min/max |
| `arrival_quantity` | float | No | Reported arrival volume | Must have source-specific unit |
| `rainfall_mm` | float | No | Daily rain | Non-negative after source QA |
| `temp_min_c` | float | No | Daily minimum temperature | Plausibility check required |
| `temp_max_c` | float | No | Daily maximum temperature | Must be ≥ minimum |
| `latitude` | float | No | Market latitude | −90 to 90 |
| `longitude` | float | No | Market longitude | −180 to 180 |
| `is_synthetic` | boolean | Demo only | Prevents fake performance claims | Must be false/absent for real evaluation |

## Engineered fields

| Feature family | Examples | Leakage rule |
|---|---|---|
| Calendar | month, weekday, week, cyclical day-of-year | Known at prediction time |
| Price lags | 1, 7, 14, 28 days | Same market/commodity/variety only |
| Rolling price | mean/std over 7, 14, 28 days | Always shifted by one day |
| Arrivals | lag 1, shifted rolling mean 7 | Used only if real source exists |
| Spread | max price − min price | Same-day published input |
| Target return | future/current − 1 | Target only; never a feature |

## Unique key

`state + district + market + commodity + variety + arrival_date`

Conflicts are retained in the raw layer and resolved by a documented rule. The current cleaned layer keeps the final reported duplicate.

