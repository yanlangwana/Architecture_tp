import os

import pytest

os.environ["FRAUDLAKE_SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from serving.app import app  # noqa: E402
from serving.schemas import EXAMPLE  # noqa: E402


class FakeScorer:
    """Remplace le PipelineModel Spark : score = 0.95 si montant élevé, sinon 0.02."""

    ready, error, version, threshold = True, None, "7", 0.9
    metadata = {"model_version": "7", "threshold": 0.9}

    def score(self, rows):
        return [0.95 if r["amt_log"] > 6 else 0.02 for r in rows]


@pytest.fixture()
def client():
    app.state.scorer = FakeScorer()
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "model_version": "7", "detail": None}


def test_predict_alert_with_reasons(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "ALERTE" and body["priority"] == "élevée" and body["model_version"] == "7"
    assert len(body["reasons"]) >= 3


def test_predict_accepted_transaction(client):
    body = client.post("/predict", json={**EXAMPLE, "amt": 25.0, "amt_sum_24h": 80, "amt_ratio_to_card_mean": 0.8}).json()
    assert body["decision"] == "ACCEPTÉE" and body["reasons"] == []


@pytest.mark.parametrize("patch", [{"amt": -5}, {"hour": 25}, {"category": "casino"}, {"gender": "X"}])
def test_invalid_payload_is_rejected(client, patch):
    assert client.post("/predict", json={**EXAMPLE, **patch}).status_code == 422


def test_batch_and_stats(client):
    r = client.post("/predict/batch", json={"transactions": [EXAMPLE, {**EXAMPLE, "amt": 12.0}]})
    assert r.status_code == 200 and r.json()["n_alerts"] == 1
    assert client.get("/stats").json()["transactions"] >= 2


def test_service_unavailable_without_model():
    class NotReady(FakeScorer):
        ready, error = False, "pas de champion"

    app.state.scorer = NotReady()
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=EXAMPLE).status_code == 503
