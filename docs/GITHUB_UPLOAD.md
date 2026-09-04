# Review and upload to GitHub

Your repository is already initialised on GitHub. Review this project locally before pushing it.

## 1. Extract and inspect

Extract the provided ZIP. Start with `README.md`, then follow `docs/LEARNING_AND_REVIEW_GUIDE.md`. Do not add your API key to any file.

## 2. Test from the project directory

```bash
python -m pip install -e ".[dev,app]"
python -m agridecision.cli demo --output-dir artifacts
python -m pytest
python -m ruff check src tests
python scripts/verify_release.py
```

The demo metrics are synthetic and cannot be used as LinkedIn performance claims.

## 3. Clone your existing repository

```bash
git clone https://github.com/ashutoshmahanta714-bit/agridecision-os-india.git
cd agridecision-os-india
```

Copy the extracted project contents into this cloned folder. Keep the hidden `.git` directory created by `git clone`.

## 4. Review exactly what will be published

```bash
git status
git add .
git diff --cached
```

Confirm that none of these appear in the staged files:

- `.env`;
- your Data.gov.in API key;
- raw downloaded data;
- trained `.joblib` files;
- generated predictions or metrics;
- private information.

If a secret was ever committed, revoke it immediately. Removing it only in a later commit does not remove it from Git history.

## 5. Commit and push

```bash
git commit -m "Build AgriDecision OS v0.1"
git push origin main
```

Open the GitHub **Actions** tab and confirm that lint, tests, and the 70% coverage gate pass.

## 6. Real-data milestone

After obtaining a historical official snapshot:

```bash
python -m agridecision.cli prepare --input data/raw/mandi_prices.csv --output-dir data/processed
python -m agridecision.cli train --input data/processed/supervised_features.csv --output-dir artifacts
```

Check `training_readiness.json`, `data_quality_report.json`, `snapshot_manifest.json`, real chronological metrics, market slices, and failure examples before creating a LinkedIn post.

