import pytest
from fastapi.testclient import TestClient

from triage import config, train


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "ARTIFACT_DIR", out)
    mp.setattr(config, "MODEL_PATH", out / "model.joblib")
    mp.setattr(config, "METRICS_PATH", out / "metrics.json")
    train.main()
    yield out
    mp.undo()


@pytest.fixture(scope="session")
def client(artifacts):
    from triage.api import app

    with TestClient(app) as c:
        yield c
