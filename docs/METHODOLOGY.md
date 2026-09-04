# Modelling and evaluation methodology

## Why random train/test split is prohibited

Random splitting allows future market regimes to influence training and overstates operational performance. The repository keeps the newest dates as the untouched test period. An earlier chronological validation period selects the shock decision threshold and conformal interval radius.

## Models

- **Seasonal naive:** seven-day lag when available. This is the minimum credible forecasting baseline.
- **Forecast:** gradient-boosted trees over encoded market identity, calendar, lag, rolling, weather, arrival, and geospatial features.
- **Risk:** regularised class-weighted logistic regression. Probability quality matters more than raw accuracy.
- **Anomaly:** Isolation Forest trained only on the training period.

Model complexity should increase only when rolling-origin evidence beats the baseline across enough markets.

## Metrics

Forecast: MAE, RMSE, MAPE, R², and MASE. MASE below 1 means the forecast beats the selected naive scaling error.

Risk: precision, recall, F1, ROC-AUC, PR-AUC, and Brier score. PR-AUC is essential when shocks are rare. Accuracy is intentionally not the headline metric.

Uncertainty: empirical coverage and interval width for nominal 90% split-conformal intervals.

Decision: expected net price/value and regret in simulation. Offline prediction quality and decision value are different questions.

## Leakage audit

- Target price and target return are excluded from features.
- Rolling features use `shift(1)`.
- Target is joined on the exact future date.
- Threshold and uncertainty calibration exclude the final test period.
- Preprocessors are fitted inside pipelines on training rows.

## Real-data experiment protocol

1. Freeze and checksum the source snapshot.
2. Document missing dates and reporting changes.
3. Reserve the newest period once.
4. Run seasonal naive and recent-median baselines.
5. Perform rolling-origin validation for candidate models.
6. Lock hyperparameters.
7. Evaluate once on the test period.
8. Publish all metrics, slices, failure examples, and limitations.

