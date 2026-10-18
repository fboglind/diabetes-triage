from triage import config

SAMPLE = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": config.MODEL_VERSION}


def test_predict_returns_float(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction"}
    assert isinstance(body["prediction"], float)


def test_missing_feature_returns_422(client):
    payload = {k: v for k, v in SAMPLE.items() if k != "s6"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "s6"]


def test_wrong_type_returns_422(client):
    r = client.post("/predict", json={**SAMPLE, "bmi": "abc"})
    assert r.status_code == 422


def test_unknown_field_returns_422(client):
    r = client.post("/predict", json={**SAMPLE, "bmii": 0.1})
    assert r.status_code == 422


def test_key_order_does_not_matter(client):
    shuffled = dict(reversed(list(SAMPLE.items())))
    a = client.post("/predict", json=SAMPLE).json()["prediction"]
    b = client.post("/predict", json=shuffled).json()["prediction"]
    assert a == b
