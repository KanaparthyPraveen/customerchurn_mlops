from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Churn" in response.text


def test_prediction_page():
    response = client.get("/prediction")
    assert response.status_code == 200


def test_result_page():
    response = client.get("/result")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"


def test_predict_likely_churn():
    data = {
        "gender": 1,
        "senior_citizen": 0,
        "tenure": 3,
        "monthly_charges": 85.0,
        "contract": 0,
        "internet_service": 1
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "probability" in payload
    assert "risk_percentage" in payload
    assert "risk_level" in payload
    assert "risk_factors" in payload
    assert "recommendations" in payload
    assert payload["prediction"] in [0, 1]


def test_predict_unlikely_churn():
    data = {
        "gender": 0,
        "senior_citizen": 0,
        "tenure": 60,
        "monthly_charges": 25.0,
        "contract": 2,
        "internet_service": 0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 0
    assert payload["risk_percentage"] < 50.0
