import base64
import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def encode(value: object) -> str:
    raw = json.dumps(value).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def sample_payload(priority: str = "HIGH") -> str:
    return encode(
        {
            "adaptInput": {
                "user": {"id": "U42", "fullName": "Jane Doe"},
                "action": "CREATE",
                "metadata": {"priority": priority},
            }
        }
    )


def phase_2_payload() -> str:
    return encode(
        {
            "adaptInput": {
                "user": {"id": "U42", "fullName": "Jane Doe"},
                "action": "CREATE",
                "metadata": {"priority": "HIGH"},
            },
            "heartbeats": [
                {
                    "service": "auth",
                    "timestamp": 1710000123,
                    "latencyMs": 120,
                    "status": "OK",
                },
                {
                    "service": "auth",
                    "timestamp": 1710000125,
                    "latencyMs": 180,
                    "status": "FAIL",
                },
                {
                    "service": "auth",
                    "timestamp": 1710000121,
                    "latencyMs": 95,
                    "status": "OK",
                },
            ],
            "sloQuery": {"service": "auth", "since": 1710000123},
        }
    )


def test_legacy_and_namespaced_routes_transform_the_sample() -> None:
    expected = {
        "adaptOutput": {
            "id": "U42",
            "name": "Jane Doe",
            "action": "create",
            "priority": 3,
        }
    }

    for route in ("/solve", "/adaptive-api/solve"):
        response = client.post(route, json={"payload": sample_payload()})
        assert response.status_code == 200
        assert response.json() == expected


def test_all_priority_values_are_mapped() -> None:
    for priority, expected in (("LOW", 1), ("MEDIUM", 2), ("HIGH", 3)):
        response = client.post("/solve", json={"payload": sample_payload(priority)})
        assert response.status_code == 200
        assert response.json()["adaptOutput"]["priority"] == expected


def test_phase_2_sample_combines_adaptation_and_slo_metrics() -> None:
    expected = {
        "adaptOutput": {
            "id": "U42",
            "name": "Jane Doe",
            "action": "create",
            "priority": 3,
        },
        "sloOutput": {
            "availability": 0.5,
            "p95LatencyMs": 180,
        },
    }

    for route in ("/solve", "/adaptive-api/solve"):
        response = client.post(route, json={"payload": phase_2_payload()})
        assert response.status_code == 200
        assert response.json() == expected


def test_slo_filters_service_and_uses_inclusive_since_boundary() -> None:
    payload = encode(
        {
            "adaptInput": {
                "user": {"id": "U1", "fullName": "A"},
                "action": "UPDATE",
                "metadata": {"priority": "LOW"},
            },
            "heartbeats": [
                {
                    "service": "auth",
                    "timestamp": 100,
                    "latencyMs": 10,
                    "status": "FAIL",
                },
                {
                    "service": "billing",
                    "timestamp": 101,
                    "latencyMs": 999,
                    "status": "FAIL",
                },
                {
                    "service": "auth",
                    "timestamp": 99,
                    "latencyMs": 888,
                    "status": "FAIL",
                },
                {
                    "service": "auth",
                    "timestamp": 102,
                    "latencyMs": 20,
                    "status": "OK",
                },
            ],
            "sloQuery": {"service": "auth", "since": 100},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.status_code == 200
    assert response.json()["sloOutput"] == {
        "availability": 0.5,
        "p95LatencyMs": 20,
    }


def test_p95_uses_nearest_rank_and_is_independent_of_input_order() -> None:
    latencies = list(range(20, 0, -1))
    payload = encode(
        {
            "adaptInput": {
                "user": {"id": "U2", "fullName": "B"},
                "action": "READ",
                "metadata": {"priority": "MEDIUM"},
            },
            "heartbeats": [
                {
                    "service": "search",
                    "timestamp": index,
                    "latencyMs": latency,
                    "status": "OK",
                }
                for index, latency in enumerate(latencies, start=1)
            ],
            "sloQuery": {"service": "search", "since": 1},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.status_code == 200
    assert response.json()["sloOutput"] == {
        "availability": 1.0,
        "p95LatencyMs": 19,
    }


def test_empty_slo_window_returns_zero_metrics() -> None:
    payload = encode(
        {
            "adaptInput": {
                "user": {"id": "U3", "fullName": "C"},
                "action": "DELETE",
                "metadata": {"priority": "HIGH"},
            },
            "heartbeats": [],
            "sloQuery": {"service": "missing", "since": 500},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.status_code == 200
    assert response.json()["sloOutput"] == {
        "availability": 0.0,
        "p95LatencyMs": 0,
    }


def test_partial_slo_input_is_rejected() -> None:
    decoded = {
        "adaptInput": {
            "user": {"id": "U4", "fullName": "D"},
            "action": "READ",
            "metadata": {"priority": "LOW"},
        },
        "heartbeats": [],
    }

    response = client.post("/solve", json={"payload": encode(decoded)})

    assert response.status_code == 422


def test_invalid_base64_returns_400() -> None:
    response = client.post("/solve", json={"payload": "not valid base64!"})
    assert response.status_code == 400
    assert response.json() == {"detail": "payload must be valid Base64"}


def test_invalid_decoded_model_returns_422() -> None:
    response = client.post("/solve", json={"payload": encode({"wrong": "shape"})})
    assert response.status_code == 422
