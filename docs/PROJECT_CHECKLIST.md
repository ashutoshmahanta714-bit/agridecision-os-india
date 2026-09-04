# Project checklist

## Core release

- [x] Problem, user, decision, unit of observation, and targets defined
- [x] Secure API key handling
- [x] API pagination, retry, rate-limit handling, checkpoint, resume
- [x] Official CSV/ZIP fallback
- [x] Canonical schema, quality quarantine, and readiness gate
- [x] Snapshot checksum and provenance manifest
- [x] SQL schema, analytical queries, and local SQLite store
- [x] Leakage-safe feature engineering
- [x] Chronological train/calibration/test logic
- [x] Two simple forecast baselines
- [x] Regression, classification, anomaly, and uncertainty models
- [x] Market-level evaluation and monitoring
- [x] Ranking, optimisation, and sequential-decision simulation
- [x] API, dashboard, Docker, CI, notebooks, tests, documentation

## Real-data release

- [ ] Obtain and freeze an official historical snapshot
- [ ] Verify date range, update policy, fields, units, licence, and checksum
- [ ] Identify eligible mandis from `training_readiness.json`
- [ ] Complete EDA and missing-date analysis
- [ ] Lock train/calibration/test windows
- [ ] Tune with rolling-origin validation
- [ ] Evaluate untouched test period once
- [ ] Run market slices and failure analysis
- [ ] Conduct domain-expert and responsible-use review
- [ ] Replace demo screenshots with labelled real-data evidence
- [ ] Publish LinkedIn post and live demo only after security review

## Optional modality gates

- [ ] Verify arrival-quantity source and unit
- [ ] Join weather with source-time-safe availability
- [ ] Build dated/labelled bulletin corpus
- [ ] Build licensed image dataset and label protocol
- [ ] Validate satellite geometry, cloud mask, and acquisition timing
- [ ] Demonstrate incremental out-of-time value for each modality

