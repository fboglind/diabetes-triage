import json

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triage import config


def load_data():
    frame = load_diabetes(as_frame=True).frame
    return frame[config.FEATURES], frame["target"]


def kfold():
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def build_baseline():
    return make_pipeline(StandardScaler(), LinearRegression())


def build_pipeline():
    return make_pipeline(StandardScaler(), LassoCV(cv=kfold()))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_rmse(pipeline, X, y):
    scores = -cross_val_score(pipeline, X, y, cv=kfold(), scoring="neg_root_mean_squared_error")
    return {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}


def calibrate_threshold(pipeline, X_train, y_train, cutoff):
    """Highest threshold whose out-of-fold recall on the training set >= TARGET_RECALL."""
    oof = cross_val_predict(pipeline, X_train, y_train, cv=kfold())
    _, recall, thresholds = precision_recall_curve(y_train >= cutoff, oof)
    ok = np.where(recall[:-1] >= config.TARGET_RECALL)[0]
    return float(thresholds[ok.max()])


def flag_metrics(y_true, y_pred, cutoff, threshold):
    truth, flagged = y_true >= cutoff, y_pred >= threshold
    return {
        "threshold": round(threshold, 2),
        "precision": round(float(precision_score(truth, flagged, zero_division=0)), 4),
        "recall": round(float(recall_score(truth, flagged)), 4),
    }


def main():
    np.random.seed(config.SEED)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED
    )
    cutoff = float(np.quantile(y_train, config.HIGH_RISK_QUANTILE))

    baseline = build_baseline().fit(X_train, y_train)
    pipeline = build_pipeline()
    threshold = calibrate_threshold(pipeline, X_train, y_train, cutoff)
    pipeline.fit(X_train, y_train)

    base_pred, pred = baseline.predict(X_test), pipeline.predict(X_test)
    lasso = pipeline[-1]

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_version": config.MODEL_VERSION,
            "features": config.FEATURES,
            "high_risk_threshold": threshold,
        },
        config.MODEL_PATH,
    )

    metrics = {
        "model_version": config.MODEL_VERSION,
        "model": "StandardScaler + LassoCV",
        "lasso_alpha": round(float(lasso.alpha_), 4),
        "features_used": int(np.sum(lasso.coef_ != 0)),
        "rmse_test": round(rmse(y_test, pred), 4),
        "rmse_cv_train": cv_rmse(build_pipeline(), X_train, y_train),
        "high_risk_cutoff": round(cutoff, 2),
        "flag_test": {
            "naive": flag_metrics(y_test, pred, cutoff, cutoff),
            "calibrated": flag_metrics(y_test, pred, cutoff, threshold),
        },
        "baseline_v0.1": {
            "model": "StandardScaler + LinearRegression",
            "rmse_test": round(rmse(y_test, base_pred), 4),
            "rmse_cv_train": cv_rmse(build_baseline(), X_train, y_train),
            "flag_test_naive": flag_metrics(y_test, base_pred, cutoff, cutoff),
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_test_high_risk": int(np.sum(y_test >= cutoff)),
        "seed": config.SEED,
        "sklearn_version": sklearn.__version__,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
