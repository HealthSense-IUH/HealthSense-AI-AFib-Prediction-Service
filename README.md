# HealthSense AI AFib Prediction Service

FastAPI service for PPG-based suspected-AF screening. The default model is
**V6C stacking**: ExtraTrees, RandomForest and XGBoost feed a StandardScaler +
LogisticRegression meta model. Inference returns the raw meta probability;
the research calibrator and temporal alert rules are not enabled.

## Models

Place both files in `app/models/` when running locally:

- `v6_stacking.pkl` (approximately 1.88 GB).
- `v6_stacking.json` (matching model card and SHA-256).

The service defaults to `MODEL_FILE=v6_stacking.pkl`. Legacy `v4.pkl` remains
supported when explicitly selected. Model switches validate the feature schema
and preprocessing; a failed switch preserves the active model. V6 loading
checks the model hash and card feature order before activating the bundle.

The service extracts HRV features with the existing signal/SQI pipeline and
adds `PPG_AC`: the mean absolute normalized autocorrelation over non-zero lags,
matching the V6 training implementation. V6 consumes the 14 features declared
in its bundle/card.

## Local use

```bash
python -m venv venv
# Activate the virtual environment, then:
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Download the two model files manually into `app/models/`, or set
`GOOGLE_DRIVE_MODEL_URL` and `GOOGLE_DRIVE_MODEL_CARD_URL` in your environment:

```bash
python -m pip install 'gdown==5.2.0' 'google-auth[requests]>=2.0,<3.0'
python scripts/download_drive_models.py
python -m app.services.verify_model
```

The download dependencies are CI tooling and are not required by the running service.

## CI/CD

On a push to `main`, the Docker job downloads the model pair from Google Drive
into `app/models/`, verifies it, then builds and pushes the Docker image to
Docker Hub. The image includes both files and defaults to V6. A Docker build
smoke check loads the actual model and predicts from synthetic PPG; failure
stops image publication.

Configure repository Actions secrets:

| Secret | Value |
|--------|-------|
| `GOOGLE_DRIVE_MODEL_URL` | Drive share link for `v6_stacking.pkl` |
| `GOOGLE_DRIVE_MODEL_CARD_URL` | Drive share link for `v6_stacking.json` |
| `GOOGLE_DRIVE_CREDENTIALS` | Optional Google service account JSON key for private files |
| `DOCKERHUB_USERNAME` | Docker Hub account |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

Public files need **Anyone with the link / Viewer** access. For private files,
enable Google Drive API for the service account, share both files with its
email as Viewer, and configure `GOOGLE_DRIVE_CREDENTIALS`.

GitHub Actions caches the pair and verifies the pinned model SHA-256 and card
on every build. Access errors, incomplete downloads and checksum mismatches
stop the job. For a new artifact, update the link secrets, checksum in
`scripts/download_drive_models.py`, and workflow cache key together.

Publishing an image does not replace the running server. Deploy the new image
and check `GET /api/health`: `active_model_file` should be `v6_stacking.pkl`,
`model_version` should identify the V6 research freeze, and `feature_count`
should be 14. Remove any old `MODEL_FILE=v4.pkl` override from deployment
environment variables or `.env` if V6 is intended.

## API

Swagger: `/docs`.

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/` | Service version |
| GET | `/api/health` | Active model/version/features; HTTP 503 if no model is loaded |
| GET | `/api/models` | List model files and card metadata without loading inactive models into RAM |
| POST | `/api/models/active` | Select a validated model from `app/models` |
| POST | `/api/predict` | RR intervals plus `ppg_ac` for V6 |
| POST | `/api/predict-csv` | Raw PPG CSV; computes all V6 features automatically |

CSV columns: `Time(ms)` + `IR`, or `time` + `ppg`. RabbitMQ/S3 processing
also computes `PPG_AC` automatically from the recording.

RR intervals alone cannot reconstruct PPG autocorrelation. A V6 request to
`/api/predict` must provide `ppg_ac` computed from the same raw PPG window;
otherwise it returns HTTP 422 instead of substituting zero. Workout-muted
requests keep their existing behavior.

## Checks

```bash
ruff check .
ruff format --check .
python -m unittest discover -s tests -p 'test_v6.py'
python -m app.services.verify_model
```

The Docker smoke check verifies loading and execution, not clinical performance.
Research evaluation and scope are described in the model card.
