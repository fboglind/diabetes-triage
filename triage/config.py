import os
from pathlib import Path

FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_VERSION = "v0.2"

# High-risk flag: "high risk" = top 25% of progression in the training data.
HIGH_RISK_QUANTILE = 0.75
TARGET_RECALL = 0.80

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
