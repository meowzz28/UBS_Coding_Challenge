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


def test_identity_shift_and_missingness_raise_risk_only_on_a_connected_flow() -> None:
    reset()
    consistent = process(
        tx("consistent-1", "a", "b", 0, deviceId="device-one"),
        tx("consistent-2", "b", "c", 1, deviceId="device-one"),
    )[-1]["riskScore"]

    reset()
    shifted = process(
        tx("shift-1", "a", "b", 0, deviceId="device-one"),
        tx("shift-2", "b", "c", 1, deviceId="device-two"),
    )[-1]["riskScore"]

    reset()
    missing_mid_flow = process(
        tx("missing-1", "a", "b", 0, deviceId="device-one"),
        tx("missing-2", "b", "c", 1),
    )[-1]["riskScore"]

    reset()
    isolated_missing = process(tx("isolated-missing", "x", "y", 0))[-1]["riskScore"]

    assert shifted > consistent
    assert missing_mid_flow > consistent
    assert isolated_missing == 0.02


def test_missing_identity_remains_visible_beyond_the_immediately_previous_leg() -> None:
    reset()
    process(
        tx("trail-1", "a", "b", 0, deviceId="device-one"),
        tx("trail-2", "b", "c", 1),
    )
    continuing_without_identity = process(tx("trail-3", "c", "d", 2))[0]["riskScore"]

    reset()
    ordinary_extension = process(
        tx("ordinary-1", "a", "b", 0),
        tx("ordinary-2", "b", "c", 1),
        tx("ordinary-3", "c", "d", 2),
    )[-1]["riskScore"]

    assert continuing_without_identity > ordinary_extension


def test_ip_and_device_are_independent_identity_dimensions() -> None:
    reset()
    one_dimension_changes = process(
        tx(
            "single-change-1",
            "a",
            "b",
            0,
            ipAddress="192.0.2.1",
            deviceId="device-one",
        ),
        tx(
            "single-change-2",
            "b",
            "c",
            1,
            ipAddress="192.0.2.1",
            deviceId="device-two",
        ),
    )[-1]["riskScore"]

    reset()
    both_dimensions_change = process(
        tx(
            "double-change-1",
            "a",
            "b",
            0,
            ipAddress="192.0.2.1",
            deviceId="device-one",
        ),
        tx(
            "double-change-2",
            "b",
            "c",
            1,
            ipAddress="192.0.2.2",
            deviceId="device-two",
        ),
    )[-1]["riskScore"]

    assert both_dimensions_change > one_dimension_changes


def test_shared_identity_across_disconnected_components_is_a_bounded_hint() -> None:
    reset()
    unrelated_scores = process(
        tx("unrelated-1", "a", "b", 0, ipAddress="192.0.2.1"),
        tx("unrelated-2", "c", "d", 1, ipAddress="192.0.2.2"),
        tx("unrelated-3", "e", "f", 2, ipAddress="192.0.2.3"),
    )

    reset()
    shared_scores = process(
        tx("shared-1", "a", "b", 0, ipAddress="192.0.2.1"),
        tx("shared-2", "c", "d", 1, ipAddress="192.0.2.1"),
        tx("shared-3", "e", "f", 2, ipAddress="192.0.2.1"),
    )

    assert shared_scores[1]["riskScore"] > unrelated_scores[1]["riskScore"]
    assert shared_scores[2]["riskScore"] > shared_scores[1]["riskScore"]
    assert shared_scores[2]["riskScore"] < 0.20


def test_identity_state_obeys_the_exact_24_hour_boundary() -> None:
    reset()
    first = tx("boundary-first", "a", "b", 0, deviceId="device-one")
    exactly_24_hours = tx("boundary-second", "b", "c", 0)
    exactly_24_hours["createdAt"] = (BASE_TIME + timedelta(hours=24)).isoformat()
    boundary_score = process(first, exactly_24_hours)[-1]["riskScore"]

    reset()
    process(tx("expired-first", "a", "b", 0, deviceId="device-one"))
    advance = tx("advance", "x", "y", 0)
    advance["createdAt"] = (BASE_TIME + timedelta(hours=24, seconds=1)).isoformat()
    process(advance)
    after_expiry = tx("expired-second", "b", "c", 0)
    after_expiry["createdAt"] = (BASE_TIME + timedelta(hours=24, seconds=2)).isoformat()
    expired_score = process(after_expiry)[0]["riskScore"]

    assert boundary_score > expired_score
    assert expired_score == 0.02


def test_idempotency_survives_graph_expiry_without_reintroducing_the_edge() -> None:
    reset()
    original = tx("old-id", "a", "b", 0, deviceId="device-one")
    original_result = process(original)[0]

    advance = tx("new-watermark", "x", "y", 0)
    advance["createdAt"] = (BASE_TIME + timedelta(hours=25)).isoformat()
    process(advance)
    duplicate_result = process(original)[0]

    reverse = tx("reverse-after-expiry", "b", "a", 0)
    reverse["createdAt"] = (BASE_TIME + timedelta(hours=25, minutes=1)).isoformat()
    reverse_score = process(reverse)[0]["riskScore"]

    assert duplicate_result == original_result
    assert reverse_score == 0.02


def test_conflicting_identifier_inside_a_batch_is_atomic() -> None:
    reset()
    response = client.post(
        "/ghost-chains/transactions",
        json={
            "transactions": [
                tx("batch-conflict", "a", "b", 0),
                tx("batch-conflict", "a", "c", 1),
            ]
        },
    )
    assert response.status_code == 409

    # The rejected batch must not have left the first edge behind.
    reverse_score = process(tx("after-conflict", "b", "a", 2))[0]["riskScore"]
    assert reverse_score == 0.02


def test_stale_out_of_order_transaction_does_not_reenter_active_state() -> None:
    reset()
    watermark = tx("watermark", "x", "y", 0)
    watermark["createdAt"] = (BASE_TIME + timedelta(hours=48)).isoformat()
    process(watermark)

    process(tx("stale", "a", "b", 0, deviceId="device-one"))
    reverse = tx("after-stale", "b", "a", 0)
    reverse["createdAt"] = (BASE_TIME + timedelta(hours=48, minutes=1)).isoformat()
    assert process(reverse)[0]["riskScore"] == 0.02


def test_reset_clears_identity_indices_as_well_as_graph_edges() -> None:
    reset()
    process(tx("identity-before", "a", "b", 0, ipAddress="192.0.2.1"))
    reset()
    score = process(tx("identity-after", "c", "d", 1, ipAddress="192.0.2.1"))[0][
        "riskScore"
    ]
    assert score == 0.02
