# HealthSense AI AFib Prediction Service

FastAPI service for PPG-based suspected-AF screening. The default model is
**V6C stacking**: ExtraTrees, RandomForest and XGBoost feed a StandardScaler +
LogisticRegression meta model. Each 30 s window returns the raw meta probability;
the temporal persistence rule turns the window results into an alert decision.
The research Platt calibrator is not enabled.

## Processing

The RabbitMQ consumer and `POST /api/predict-csv` prepare a recording with the
same function, `app/services/recording.py`:

1. convert the time column to milliseconds if it is in seconds;
2. estimate the sampling rate from the first 100 samples;
3. drop the first 2 s (sensor start-up transient). A 72 s recording from the
   mobile app leaves 70 s.

`app/services/windowing.py` then splits the signal into 30 s windows with a
10 s step (five windows for 70 s). Each window passes the signal-quality gate
(spectral concentration ≥ 0.70, ≥ 10 valid beats, ≥ 80 % valid NN intervals,
heart rate 30–220 bpm) before its 14 features go to the model.

The temporal persistence rule raises an alert if either branch holds:

- P_meta ≥ 0.80 with base-learner disagreement σ ≤ 0.05 in 5 consecutive windows;
- P_meta ≥ 0.90 with σ ≤ 0.05 in 2 consecutive windows.

A rejected window resets both counters. The thresholds are the frozen values of
the research protocol (ML Lab `results/v6/FROZEN_GLOBAL_RULE_V2.json`).

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

## Computational cost

`scripts/benchmark_inference.py` loads the V6 model through `PredictionService`
(SHA-256 check against the card, then deserialisation) and runs the service
pipeline (`prepare_recording`, `analyse_recording`) on 72 s excerpts of real
recordings. Results depend on the hardware, which is recorded in the output.

Result on 140 non-overlapping 72 s excerpts (up to four per recording) of the 35
MIMIC PERform AF recordings (fingertip PPG, 125 Hz), measured on an Intel Core
i7-10850H laptop CPU (6 cores, 12 threads, 32 GB RAM, Windows 11, Python 3.15,
scikit-learn 1.9.1, XGBoost 3.4.0); see `docs/benchmark_v6_mimic_perform_af.json`:

| Quantity | Value |
|---|---|
| Model file | 1.88 GB |
| Model components in memory | Extra Trees 1.44 GB (500 trees, 18.0 M nodes), Random Forest 0.43 GB (500 trees, 5.4 M nodes), XGBoost 2.5 MB |
| Start-up (SHA-256 check + loading) | 6.0 s |
| Process memory after loading | 2.08 GB (+1.92 GB for the model) |
| Window analysis per 72 s recording (5 windows: quality gate, features, model, rule) | median 0.94 s, 95th percentile 1.05 s |
| Model call per window | median 164 ms, 95th percentile 234 ms |
| Windows accepted by the quality gate | 699 / 700 |

Over four runs on the same machine, the median time per recording ranged from
0.94 to 1.03 s and the median model call from 164 to 184 ms; start-up took 6–7 s.

The timing excludes the S3 download, the whole-recording display features and
the callback to the Core Service. MIMIC PERform AF is part of the training
data, so the benchmark does not report alert decisions: they would not be a
performance estimate. Deployments need about 2 GB of memory per worker for the
model.

```bash
pip install -r requirements-dev.txt
python scripts/benchmark_inference.py --model app/models/v6_stacking.pkl \n    --card app/models/v6_stacking.json \n    --recordings <ml-lab>/data/raw/mimic_perform/af <ml-lab>/data/raw/mimic_perform/non-af \n    --time-column Time --ppg-column PPG --out benchmark.json
```

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
python -m unittest discover -s tests -p 'test_*.py'
python -m app.services.verify_model
```

`tests/test_parity_v6.py` compares feature extraction with the ML Lab code. It
needs a checkout of HealthSense-MachineLearning-Lab next to this repository, or
`ML_LAB_PATH`; otherwise it is skipped locally. CI runs it with the checkout.

The Docker smoke check verifies loading and execution, not clinical performance.
Research evaluation and scope are described in the model card.
