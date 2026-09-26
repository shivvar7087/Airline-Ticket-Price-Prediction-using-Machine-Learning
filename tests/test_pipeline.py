"""
Unit and integration tests for Airline Ticket Price Prediction system.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app import app
from backend.predict import predictor

client = TestClient(app)

def test_model_artifacts_exist():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "flight_price_model.joblib")
    metrics_path = os.path.join(base_dir, "models", "metrics.json")
    metadata_path = os.path.join(base_dir, "models", "feature_metadata.json")

    assert os.path.exists(model_path), "Model pipeline file must exist"
    assert os.path.exists(metrics_path), "Metrics file must exist"
    assert os.path.exists(metadata_path), "Metadata file must exist"

def test_model_performance_threshold():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metrics_path = os.path.join(base_dir, "models", "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    r2 = metrics["test_r2_score"]
    assert r2 >= 0.85, f"R2 score should be >= 0.85, got {r2}"
    assert metrics["test_mae"] < 1000, f"MAE should be < 1000, got {metrics['test_mae']}"

def test_predictor_direct_inference():
    result = predictor.predict(
        airline="IndiGo",
        source="Delhi",
        destination="Cochin",
        date_of_journey="2026-06-15",
        dep_time="10:30",
        arrival_time="14:15",
        total_stops=1,
        additional_info="No info"
    )
    assert result["success"] is True
    assert result["predicted_price"] > 2000
    assert result["currency"] == "INR"
    assert result["duration_mins"] > 0

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_api_metadata():
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.json()
    assert len(data["airlines"]) > 0
    assert "Delhi" in data["sources"]
    assert "Cochin" in data["destinations"]

def test_api_prediction():
    res = client.post("/predict", json={
        "airline": "Air India",
        "source": "Kolkata",
        "destination": "Banglore",
        "date_of_journey": "2026-07-01",
        "dep_time": "05:50",
        "arrival_time": "13:15",
        "total_stops": 2,
        "additional_info": "No info"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["predicted_price"] > 3000
    assert "INR" in data["currency"]

if __name__ == "__main__":
    print("Running tests...")
    test_model_artifacts_exist()
    test_model_performance_threshold()
    test_predictor_direct_inference()
    test_api_health()
    test_api_metadata()
    test_api_prediction()
    print("All tests passed successfully!")
