# Changelog

## [v0.2] - 2026-10-18

### What changed
- Regressor: `LinearRegression` replaced by `LassoCV` (L1 regularisation, alpha chosen by 5-fold CV on the training set). The model uses 9 of 10 features.
- New high-risk flag. "High risk" = top 25% of progression in the training data (cutoff 214). The decision threshold on the predicted score is calibrated on out-of-fold training predictions to reach at least 80% recall; the test set is not used for tuning. The threshold ships inside the model artifact and in `metrics.json`.
- `POST /predict` accepts an optional `patient_id`. Every prediction is logged to stdout as a JSON line (timestamp, patient_id, model version, features, prediction). The response format is unchanged.

### Metrics (same held-out split as v0.1: n=89, of which 21 high-risk)

| Metric | v0.1 | v0.2 |
|---|---|---|
| Model | StandardScaler + LinearRegression | StandardScaler + LassoCV |
| RMSE, held-out test | 53.85 | 53.43 |
| RMSE, 5-fold CV on train (mean ± std) | 55.39 ± 2.36 | 55.37 ± 2.25 |
| Features used | 10 | 9 |
| High-risk threshold | 214.0 (naive: same as cutoff) | 166.9 (recall-calibrated) |
| Flag precision (test) | 1.00 | 0.53 |
| Flag recall (test) | 0.29 (6 of 21) | 0.76 (16 of 21) |

### Why v0.2 is an improvement
- **Regression accuracy is on par, not better.** The RMSE gain on the held-out split (0.4) is well within the cross-validation spread, so we do not claim an accuracy improvement. Lasso gives an equally accurate, slightly simpler model.
- **Triage quality improves substantially.** With v0.1, flagging patients whose prediction exceeds the high-risk cutoff misses 15 of 21 high-risk patients, because regression predictions are shrunk toward the mean. v0.2's calibrated threshold catches 16 of 21. In this clinic, a missed deteriorating patient is far more costly than an unnecessary follow-up call, and every flag is reviewed by a nurse, so trading precision for recall is the right choice.
- Test recall (0.76) is slightly below the 0.80 target because the threshold was set on training folds, and with 21 positives one patient moves recall by about 0.05.
- **Monitoring readiness.** Logging predictions with `patient_id` allows them to be joined with observed outcomes later, which is the basis for continued model evaluation and retraining triggers.

### Privacy note
Logged `patient_id` and features are pseudonymised health data and remain personal data under the GDPR. In production these logs need access control, a defined retention period and a documented purpose.

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
