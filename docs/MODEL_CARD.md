# Model card: seven-day onion price and shock risk

## Status

Software verification complete on labelled synthetic data. Real-data training/evaluation pending an official historical snapshot. Synthetic metrics are not performance claims.

## Intended use

Research and portfolio demonstration of market-level forecasting and decision support for selected Indian mandis.

## Out-of-scope use

Guaranteed profit, automated trading, individual credit/insurance decisions, price manipulation, or unreviewed operational advice.

## Inputs and outputs

Inputs are listed in `feature_contract.json` generated during training. Outputs are seven-day price, 90% interval, shock probability/decision, and anomaly score.

## Evaluation

Newest-date chronological holdout, seasonal-naive baseline, market-level slices, regression metrics, rare-event metrics, calibration, and interval coverage.

## Retraining triggers

Investigate retraining after persistent PSI ≥ 0.25, performance degradation beyond the agreed tolerance, a schema/source change, policy regime change, new commodity, or new horizon. Retraining is not automatic until a human reviews data quality and backtest evidence.

