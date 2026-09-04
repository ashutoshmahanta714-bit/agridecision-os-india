# Interview guide

## Thirty-second explanation

“AgriDecision OS predicts onion modal prices seven calendar days ahead across Indian mandis, estimates price-shock probability, flags anomalies, and converts predictions into cost- and risk-aware market rankings. I built a resilient government-data pipeline, explicit quality quarantine, leakage-safe features, chronological backtesting, baselines, conformal intervals, an API, a dashboard, tests, and monitoring. I report synthetic and real-data results separately.”

## Questions you must be able to answer

1. Why did you define one row as market–commodity–variety–date?
2. Why is random splitting wrong here?
3. How did you ensure rolling features cannot see the target?
4. Why compare against seasonal naive?
5. What does MASE below or above one mean?
6. Why can accuracy be misleading for rare price shocks?
7. What is the difference between ROC-AUC and PR-AUC?
8. How was the probability threshold selected without touching the test set?
9. What does a 90% conformal interval guarantee, and what does it not guarantee?
10. How does the downloader handle timeouts, 429s, interruptions, and duplicates?
11. Why quarantine invalid rows rather than silently deleting them?
12. How would you detect schema, data, concept, and performance drift?
13. Why is the highest predicted market price not always the best decision?
14. What causal assumptions are required for the intervention analysis?
15. What would you change before real farmers could rely on it?

## Honest answer to “Did ML win?”

Use the current real backtest. If the baseline wins, say so and explain the next experiment. A well-designed system that discovers a simple baseline is stronger evidence than an inflated notebook accuracy.

