from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
BASE_TIME = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


def tx(
    tx_id: str,
    source: str,
    destination: str,
    minute: int,
    **extra: object,
) -> dict[str, object]:
    return {
        "txId": tx_id,
        "fromUserId": source,
        "toUserId": destination,
        "amount": 100.0,
        "createdAt": (BASE_TIME + timedelta(minutes=minute)).isoformat(),
        **extra,
    }


def reset() -> None:
    response = client.post("/ghost-chains/reset", json={"clearTransactions": True})
    assert response.status_code == 200


def process(*transactions: dict[str, object]) -> list[dict[str, object]]:
    response = client.post(
        "/ghost-chains/transactions",
        json={"transactions": list(transactions)},
    )
    assert response.status_code == 200, response.text
    return response.json()["transactions"]


def test_required_health_and_reset_endpoints() -> None:
    assert client.get("/ghost-chains/health").json() == {"status": "ok"}
    response = client.post("/ghost-chains/reset", json={"clearTransactions": True})
    assert response.json() == {"clearTransactions": True}


def test_batch_order_is_preserved_and_scores_stay_in_range() -> None:
    reset()
    transactions = [
        tx("one", "meridian", "apex", 0),
        tx("two", "apex", "cascade", 1),
        tx("three", "cascade", "meridian", 2),
    ]
    results = process(*transactions)

    assert [result["txId"] for result in results] == ["one", "two", "three"]
    assert all(0.0 <= result["riskScore"] <= 1.0 for result in results)
    assert results[0]["riskScore"] < results[1]["riskScore"] < results[2]["riskScore"]


def test_examples_have_the_required_structural_ordering() -> None:
    reset()
    isolated = process(tx("i1", "meridian", "apex", 0))[-1]["riskScore"]

    reset()
    extension = process(
        tx("e1", "meridian", "apex", 0),
        tx("e2", "apex", "cascade", 1),
    )[-1]["riskScore"]

    reset()
    convergence = process(
        tx("c1", "meridian", "apex", 0),
        tx("c2", "meridian", "horizon", 1),
        tx("c3", "apex", "sterling", 2),
        tx("c4", "horizon", "sterling", 3),
    )[-1]["riskScore"]

    reset()
    single_return = process(
        tx("r1", "meridian", "apex", 0),
        tx("r2", "apex", "cascade", 1),
        tx("r3", "cascade", "oakridge", 2),
        tx("r4", "oakridge", "apex", 3),
    )[-1]["riskScore"]

    reset()
    multiple_return = process(
        tx("m1", "meridian", "apex", 0),
        tx("m2", "apex", "cascade", 1),
        tx("m3", "cascade", "meridian", 2),
        tx("m4", "apex", "nimbus", 3),
        tx("m5", "nimbus", "meridian", 4),
    )[-1]["riskScore"]

    assert isolated < extension < convergence < single_return < multiple_return


def test_identical_duplicate_returns_original_score_without_error() -> None:
    reset()
    original = tx("same-id", "a", "b", 0, ipAddress="192.0.2.1")
    first = process(original)[0]
    duplicate = process(original)[0]
    assert duplicate == first


def test_conflicting_duplicate_returns_409() -> None:
    reset()
    process(tx("same-id", "a", "b", 0))
    response = client.post(
        "/ghost-chains/transactions",
        json={"transactions": [tx("same-id", "a", "different", 0)]},
    )
    assert response.status_code == 409


def test_unknown_and_missing_optional_fields_are_accepted() -> None:
    reset()
    result = process(tx("future", "a", "b", 0, futurePhaseField="ignored"))
    assert result[0]["txId"] == "future"


def test_transactions_older_than_24_hours_stop_influencing_scores() -> None:
    reset()
    process(tx("old", "a", "b", 0))
    after_window = tx("new", "b", "a", 0)
    after_window["createdAt"] = (BASE_TIME + timedelta(hours=24, seconds=1)).isoformat()
    score = process(after_window)[0]["riskScore"]
    assert score == 0.02


def test_reset_removes_all_graph_state() -> None:
    reset()
    process(tx("before", "a", "b", 0))
    reset()
    score = process(tx("after", "b", "a", 1))[0]["riskScore"]
    assert score == 0.02
