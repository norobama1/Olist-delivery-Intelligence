"""Smoke tests for the delivery-delay prediction API.

Run from the project root:
    pytest tests/

Requires the model bundle to exist at models/model_bundle.joblib.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from main import app  # noqa: E402

VALID_PAYLOAD = {
    "estimated_days": 6.0,
    "approval_hours": 0.4,
    "purchase_dayofweek": 4,
    "purchase_month": 11,
    "product_weight_g": 700.0,
    "product_volume_cm3": 6450.0,
    "order_value": 92.0,
    "freight_value": 16.25,
    "zip_distance_proxy": 24.0,
    "seller_delay_rate": 0.18,
    "is_peak_delayed_period": 1,
    "seller_state": "SP",
    "customer_state": "BA",
}


@pytest.fixture(scope="module")
def client():
    # context manager triggers lifespan startup → loads model bundle
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 14


def test_predict_valid_payload_returns_200(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "delay_probability" in body
    assert "predicted_delayed" in body
    assert "risk_band" in body
    assert "threshold" in body
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}


def test_predict_invalid_state_returns_422(client):
    bad_payload = {**VALID_PAYLOAD, "seller_state": "XX"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_high_risk_order(client):
    """An order with a high-delay seller + peak month should return elevated probability."""
    high_risk = {
        **VALID_PAYLOAD,
        "estimated_days": 2.0,
        "seller_delay_rate": 0.85,
        "purchase_month": 12,
        "is_peak_delayed_period": 1,
    }
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_delayed"] is True
    assert body["risk_band"] in {"medium", "high"}


def test_predict_batch_returns_matching_count(client):
    req = {"orders": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]}
    r = client.post("/predict/batch", json=req)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3
