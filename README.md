# Diabetes triage ML service

[![CI](https://github.com/fboglind/diabetes-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/fboglind/diabetes-triage/actions/workflows/ci.yml)
[![Release](https://github.com/fboglind/diabetes-triage/actions/workflows/release.yml/badge.svg)](https://github.com/fboglind/diabetes-triage/actions/workflows/release.yml)

A small ML service for a virtual diabetes clinic. It predicts a short-term disease progression score from patient features so that triage nurses can prioritise follow-up calls (higher score = higher risk). Trained on the scikit-learn Diabetes dataset as a stand-in for de-identified EHR data.

## Run a released image

Images are public on GHCR. Stop one version before starting the other, since both use port 80.

```bash
docker pull ghcr.io/fboglind/diabetes-triage:v0.1
docker run -it -p 80:80 ghcr.io/fboglind/diabetes-triage:v0.1

docker pull ghcr.io/fboglind/diabetes-triage:v0.2
docker run -it -p 80:80 ghcr.io/fboglind/diabetes-triage:v0.2
```

The model is trained during `docker build` and baked into the image, so no setup or network access is needed at runtime.

## API

### `GET /health`

```bash
curl http://localhost:80/health
```

```json
{"status": "ok", "model_version": "v0.2"}
```

`model_version` is read from the loaded model artifact, so it always reflects the model actually being served.

### `POST /predict`

All ten features are required and must be numbers. In v0.2, an optional string `patient_id` can be included; it is logged with the prediction but not returned.

```bash
curl -X POST http://localhost:80/predict -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

```json
{"prediction": 235.95}
```

### Error responses

Invalid input returns **HTTP 422** with a JSON body. `detail` is a list with one entry per problem:

| Field | Meaning |
|---|---|
| `type` | Machine-readable error type (`missing`, `float_parsing`, `extra_forbidden`, `json_invalid`) |
| `loc` | Where the error is: `["body", "<field>"]` |
| `msg` | Human-readable description |
| `input` | The offending input value (or the whole body for a missing field) |

Missing feature (`s6` omitted):

```json
{"detail": [{"type": "missing", "loc": ["body", "s6"], "msg": "Field required", "input": {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}}]}
```

Non-numeric value (`"bmi": "abc"`):

```json
{"detail": [{"type": "float_parsing", "loc": ["body", "bmi"], "msg": "Input should be a valid number, unable to parse string as a number", "input": "abc"}]}
```

Unknown field (e.g. a typo such as `"bmii"`) is rejected rather than silently ignored:

```json
{"detail": [{"type": "extra_forbidden", "loc": ["body", "bmii"], "msg": "Extra inputs are not permitted", "input": 0.1}]}
```

## Model versions

| Version | Model | Test RMSE | High-risk flag recall |
|---|---|---|---|
| v0.1 | StandardScaler + LinearRegression | 53.85 | 0.29 (naive threshold) |
| v0.2 | StandardScaler + LassoCV, recall-calibrated threshold | 53.43 | 0.76 |

See [CHANGELOG.md](CHANGELOG.md) for what changed, side-by-side metrics and the rationale.

## Reproduce locally

Requires Python 3.12 and Docker.

```bash
git clone https://github.com/fboglind/diabetes-triage.git
cd diabetes-triage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m triage.train                  # writes artifacts/model.joblib and artifacts/metrics.json
ruff check . && ruff format --check . && pytest -q
uvicorn triage.api:app --host 0.0.0.0 --port 8000
```

Build and test the container exactly as the release pipeline does:

```bash
docker build -t diabetes-triage:local .
docker run -d --name triage -p 80:80 diabetes-triage:local
./scripts/smoke_test.sh http://localhost:80 v0.2
docker rm -f triage
```

Training is deterministic: fixed seed (`42`) for the split and cross-validation, all dependencies pinned in `requirements.txt`, and the Python base image pinned by digest in the `Dockerfile`. Running `python -m triage.train` twice produces byte-identical artifacts.

## CI/CD

**`ci.yml`** runs on every pull request and push to `main`:
lint (`ruff`), unit and API tests (`pytest`, including an RMSE quality gate), a training smoke run, upload of the model and metrics as a workflow artifact, and a Docker build check.

**`release.yml`** runs on tags matching `v*`:
1. Fails if the tag does not match `MODEL_VERSION` in `triage/config.py`.
2. Fails if `CHANGELOG.md` has no entry for the version.
3. Builds the image (with OCI labels linking it to the source commit).
4. Runs the container and executes `scripts/smoke_test.sh` (health, valid prediction, 422 on missing feature).
5. Only then pushes the image to GHCR.
6. Publishes a GitHub Release with the CHANGELOG section and `metrics.json` extracted from the shipped image.

Release process: bump `MODEL_VERSION`, add a CHANGELOG entry, merge via PR, then `git tag -a vX.Y -m "..." && git push origin vX.Y`.

## Project structure

```
triage/
  config.py        features, seed, version, paths
  train.py         training, evaluation, threshold calibration
  api.py           FastAPI service (/health, /predict)
tests/             training and API tests
scripts/
  smoke_test.sh    acceptance tests against a running container
.github/workflows/
  ci.yml           PR/push pipeline
  release.yml      tag-triggered release pipeline
Dockerfile         multi-stage build; trains the model at build time
CHANGELOG.md
```

## Monitoring and privacy

From v0.2, every prediction is logged to stdout as a JSON line (timestamp, `patient_id`, model version, features, prediction), available via `docker logs`. The `patient_id` lets predictions be joined with observed outcomes later, which is the basis for monitoring model performance and triggering retraining. These logs contain pseudonymised health data, which remains personal data under the GDPR; a production deployment needs access control, a retention limit and a documented purpose.
