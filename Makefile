.PHONY: install demo test lint train api dashboard clean

install:
	python -m pip install -e ".[dev,app]"

demo:
	python -m agridecision.cli demo --output-dir artifacts

test:
	python -m pytest

lint:
	python -m ruff check src tests

train:
	python -m agridecision.cli train --input data/processed/mandi_prices.csv --output-dir artifacts

api:
	uvicorn agridecision.api.app:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

clean:
	python scripts/clean_generated.py

