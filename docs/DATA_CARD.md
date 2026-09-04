# Data card

## Core source

Data.gov.in resource `9ef84268-d588-465a-a308-a864a43d0070`, “Current Daily Price of Various Commodities from Various Markets (Mandi).” Record the final portal URL, retrieval timestamp, row count, date range, checksum, resource update time, and licence in the experiment log when the snapshot is downloaded.

The resource title alone does not prove that a downloaded snapshot contains enough history for forecasting. `training_readiness.json` therefore blocks modelling unless enough distinct dates and markets are present. If the portal download is only a current-day snapshot, obtain an official historical price report and document it as a separate source rather than treating rows across markets as a time series.

## Optional sources

- Open-Meteo archive for daily weather at documented market coordinates.
- Separately verified government arrival reports if the price resource lacks volume.
- Official agricultural bulletins with publication timestamps.
- Properly licensed and labelled produce photographs.
- Satellite observations with acquisition time, geometry, cloud masks, and processing version.

## Known biases

Coverage and reporting frequency vary by market, date, variety, and state. Modal price is a reported market statistic, not a transaction-level distribution. Missingness and revisions may not be random.

## Synthetic fixture

`data/demo.py` generates deterministic educational data for tests. It has designed relationships and disruptions, and therefore cannot estimate real performance or social impact.
