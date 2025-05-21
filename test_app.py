from main import app
import pytest

valid_payload = {
    "exports": 45.0,
    "health": 7.0,
    "gdpp": 15000.0,
    "child_mort": 30.0,
    "inflation": 3.5
}

invalid_payload = {
        "exports": 45.0,
        "health": 7.0,
        "gdpp": 15000.0,
        # missing "child_mort" and "inflation"
    }


@pytest.fixture
def client():
    return app.test_client()


def test_title(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Country Status Analyzer" in response.data


def test_predict_valid_input(client):
    response = client.post("/predict", json=invalid_payload)
    data = response.get_json()

    assert response.status_code == 200
    assert "label" in data
    assert "message" in data
    assert isinstance(data["label"], int)
    assert data["message"] in [
        "Highly Developed Country",
        "Least Developed Country",
        "Moderately Developed Country"
    ]


def test_predict_missing_fields(client):
    response = client.post("/predict", json=invalid_payload)
    data = response.get_json()

    assert response.status_code == 400
    assert "error" in data
    assert "Missing features" in data["error"]


def test_predict_invalid_json(client):
    # sending plain text instead of JSON
    response = client.post("/predict", data="Not a JSON")
    data = response.get_json()

    assert response.status_code == 400 or response.status_code == 500
    assert "error" in data


def test_predict_empty_input(client):
    response = client.post("/predict", json={})
    data = response.get_json()

    assert response.status_code == 400
    assert "error" in data
    assert data["error"] == "Invalid input. JSON data is required."
