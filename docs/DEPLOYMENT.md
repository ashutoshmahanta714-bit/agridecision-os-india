# Deployment

## Local API

```bash
python -m pip install -e ".[app]"
python -m agridecision.cli demo --output-dir artifacts
uvicorn agridecision.api.app:app --host 0.0.0.0 --port 8000
```

Check `/health`, inspect `/metadata`, then use `/docs` to test payloads. The prediction endpoint expects engineered features matching the saved contract; production orchestration should build those from validated recent history.

## Docker

```bash
docker compose up --build
```

API: port 8000. Dashboard: port 8501.

## Production checklist

- Use a managed secret store; never bake API keys into images.
- Keep model/data artifacts in versioned object storage rather than Git.
- Add authentication, HTTPS, request limits, structured logs, and audit IDs.
- Pin an immutable image and model version.
- Add source freshness and training-data provenance to responses.
- Log predictions without sensitive personal data.
- Test rollback before release.
- Configure drift/performance alert ownership.

