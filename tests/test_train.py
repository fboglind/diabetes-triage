import json
import math

from triage import config, train

RMSE_CEILING = 60.0


def test_training_writes_artifacts(artifacts):
    assert config.MODEL_PATH.exists()
    assert config.METRICS_PATH.exists()


def test_rmse_below_quality_gate(artifacts):
    metrics = json.loads(config.METRICS_PATH.read_text())
    assert math.isfinite(metrics["rmse_test"])
    assert metrics["rmse_test"] < RMSE_CEILING


def test_training_is_deterministic(artifacts):
    first = json.loads(config.METRICS_PATH.read_text())["rmse_test"]
    train.main()
    second = json.loads(config.METRICS_PATH.read_text())["rmse_test"]
    assert first == second


def test_calibrated_flag_improves_recall(artifacts):
    m = json.loads(config.METRICS_PATH.read_text())
    naive, calibrated = m["flag_test"]["naive"], m["flag_test"]["calibrated"]
    assert calibrated["threshold"] < naive["threshold"]
    assert calibrated["recall"] > naive["recall"]
