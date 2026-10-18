# Changelog

All notable changes to this project are documented here.

## [v0.1] - 2026-10-18

Baseline release.

### Model
- `StandardScaler` + `LinearRegression` in a single scikit-learn `Pipeline`.
- Diabetes dataset (scikit-learn), 80/20 held-out split, `random_state=42`.

### Metrics (held-out test set, n=89)

| Version | Model | RMSE |
|---|---|---|
| v0.1 | StandardScaler + LinearRegression | 53.85 |

### Service
- `GET /health` returns status and model version.
- `POST /predict` returns `{"prediction": <float>}`; invalid input returns HTTP 422 with a JSON error body.
- Model is trained during `docker build` and baked into the image.
