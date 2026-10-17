import json

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triage import config


def load_data():
    frame = load_diabetes(as_frame=True).frame
    return frame[config.FEATURES], frame["target"]


def build_pipeline():
    return make_pipeline(StandardScaler(), LinearRegression())


def main():
    np.random.seed(config.SEED)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED
    )

    pipeline = build_pipeline().fit(X_train, y_train)
    rmse = float(np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test))))

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_version": config.MODEL_VERSION,
            "features": config.FEATURES,
        },
        config.MODEL_PATH,
    )

    metrics = {
        "model_version": config.MODEL_VERSION,
        "model": "StandardScaler + LinearRegression",
        "rmse_test": round(rmse, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed": config.SEED,
        "sklearn_version": sklearn.__version__,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
