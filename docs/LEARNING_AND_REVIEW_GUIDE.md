# Code review and learning order

Review the repository in this order. Do not jump directly to the model.

## 1. Problem and contract

Read `docs/PROJECT_SPECIFICATION.md` and `docs/DATA_DICTIONARY.md`.

You should be able to explain the user, decision, unit of observation, exact seven-day target, shock label, inputs available at inference time, and why arrival quantity is optional.

## 2. Data engineering

Review:

1. `src/agridecision/data/data_gov.py`
2. `src/agridecision/data/csv_loader.py`
3. `src/agridecision/data/schema.py`
4. `src/agridecision/data/quality.py`
5. `src/agridecision/data/sqlite_store.py`
6. `sql/schema.sql` and `sql/analysis_queries.sql`

Exercises: explain pagination, HTTP 429, checkpointing, idempotence, quarantine, primary keys, and window functions.

## 3. EDA and statistics

Use `notebooks/02_eda_statistics_sql.ipynb`. Explain distributions, missingness, market coverage, outliers, confidence intervals, and why an effect size matters alongside a p-value.

## 4. Feature engineering

Review `src/agridecision/features/tabular.py`. Manually verify one row's 1-day lag, 7-day target, and 7-day rolling mean. Explain why every rolling window is shifted.

## 5. Machine learning

Review model files, then `models/training.py`. Explain pipelines, encoding, imputation, regression versus classification, imbalance, anomaly detection, temporal validation, baselines, MASE, PR-AUC, Brier score, and conformal coverage.

## 6. Decisions and monitoring

Review ranking, optimisation, drift, and delayed outcomes. Explain why the market with the highest forecast price might not have the best expected net value.

## 7. Advanced laboratories

Review one laboratory at a time using `docs/ADVANCED_LABS.md`. For each, state its data requirement, assumptions, evaluation method, and reason it could fail.

## 8. Software engineering

Run tests and inspect CI, configuration, API validation, dashboard behaviour, Docker, secrets, and `.gitignore`.

## Interview readiness test

You are ready to present the project when you can answer all questions in `docs/INTERVIEW_GUIDE.md` without reading and can reproduce the real-data pipeline from a clean environment.

