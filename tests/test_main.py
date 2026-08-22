import base64
import json

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def encode(value: object) -> str:
    raw = json.dumps(value).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def test_sample_request_is_transformed() -> None:
    payload = encode(
        {
            "adaptInput": {
                "user": {"id": "U42", "fullName": "Jane Doe"},
                "action": "CREATE",
                "metadata": {"priority": "HIGH"},
            }
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.status_code == 200
    assert response.json() == {
        "adaptOutput": {
            "id": "U42",
            "name": "Jane Doe",
            "action": "create",
            "priority": 3,
        }
    }


def test_all_priority_values_are_mapped() -> None:
    for priority, expected in (("LOW", 1), ("MEDIUM", 2), ("HIGH", 3)):
        payload = encode(
            {
                "adaptInput": {
                    "user": {"id": "U1", "fullName": "Test User"},
                    "action": "UPDATE",
                    "metadata": {"priority": priority},
                }
            }
        )
        response = client.post("/solve", json={"payload": payload})

        assert response.status_code == 200
        assert response.json()["adaptOutput"]["priority"] == expected


def test_invalid_base64_returns_400() -> None:
    response = client.post("/solve", json={"payload": "not valid base64!"})

    assert response.status_code == 400
    assert response.json() == {"detail": "payload must be valid Base64"}


def test_invalid_decoded_model_returns_422() -> None:
    response = client.post("/solve", json={"payload": encode({"wrong": "shape"})})

    assert response.status_code == 422


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
