from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.challenges.showdown import (
    decide_move,
    pre_reveal_equity,
    showdown_equity,
)
from app.main import app


def sample_state() -> dict[str, Any]:
    return {
        "protocol_version": 2,
        "match_id": "phase1-seed7",
        "phase": 1,
        "table_rule": "standard",
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": 200,
        "your_stack": 185,
        "hand_number": 6,
        "total_hands": 100,
        "round": "post_reveal",
        "your_number": 3,
        "community_number": 5,
        "your_seat": 0,
        "button_seat": 1,
        "pot": 32,
        "to_call": 18,
        "min_raise_to": 36,
        "max_raise_to": 185,
        "legal_actions": ["fold", "call", "raise"],
        "players": [
            {
                "seat": 0,
                "name": "you",
                "bet_this_round": 0,
                "stack": 185,
            },
            {
                "seat": 1,
                "name": "Gaston",
                "bet_this_round": 18,
                "stack": 183,
            },
        ],
        "current_hand_actions": [
            {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 7},
            {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 7},
            {"round": "post_reveal", "seat": 0, "action": "check"},
            {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 18},
        ],
        "recent_hands": [],
    }


def test_exact_equities_cover_pairs_high_cards_and_pre_reveal_range() -> None:
    assert showdown_equity(5, 5) == pytest.approx(12.5 / 13)
    assert showdown_equity(13, 5) == pytest.approx(11.5 / 13)
    assert showdown_equity(3, 5) == pytest.approx(2.5 / 13)
    assert pre_reveal_equity(1) == pytest.approx(18.5 / 169)
    assert pre_reveal_equity(7) == pytest.approx(0.5)
    assert pre_reveal_equity(13) == pytest.approx(150.5 / 169)


def test_documentation_example_folds_weak_number_to_large_bet() -> None:
    assert decide_move(sample_state()) == {"action": "fold"}


def test_pair_raises_inside_authoritative_range() -> None:
    state = sample_state()
    state["your_number"] = 5
    result = decide_move(state)
    assert result["action"] == "raise"
    assert state["min_raise_to"] <= result["amount"] <= state["max_raise_to"]


def test_pair_calls_when_raising_is_unavailable() -> None:
    state = sample_state()
    state["your_number"] = 5
    state["legal_actions"] = ["fold", "call"]
    state["min_raise_to"] = None
    state["max_raise_to"] = None
    assert decide_move(state) == {"action": "call"}


def test_button_completes_affordable_middle_number() -> None:
    state = sample_state()
    state.update(
        {
            "round": "pre_reveal",
            "community_number": None,
            "your_number": 7,
            "pot": 3,
            "to_call": 1,
            "min_raise_to": 4,
            "max_raise_to": 200,
            "legal_actions": ["fold", "call", "raise"],
            "current_hand_actions": [],
        }
    )
    state["players"][0]["bet_this_round"] = 1
    assert decide_move(state) == {"action": "call"}


@pytest.mark.parametrize(
    ("legal", "to_call", "minimum", "maximum"),
    [
        (["check", "bet"], 0, 2, 30),
        (["fold", "call", "raise"], 7, 14, 48),
        (["check"], 0, None, None),
        (["fold", "call"], 12, None, None),
    ],
)
def test_every_response_obeys_actions_and_amount_contract(
    legal: list[str],
    to_call: int,
    minimum: int | None,
    maximum: int | None,
) -> None:
    for number in range(1, 14):
        for community in range(1, 14):
            state = sample_state()
            state.update(
                {
                    "your_number": number,
                    "community_number": community,
                    "legal_actions": legal,
                    "to_call": to_call,
                    "min_raise_to": minimum,
                    "max_raise_to": maximum,
                }
            )
            result = decide_move(state)
            assert result["action"] in legal
            if result["action"] in {"bet", "raise"}:
                assert set(result) == {"action", "amount"}
                assert minimum is not None and maximum is not None
                assert minimum <= result["amount"] <= maximum
            else:
                assert set(result) == {"action"}


def test_decisions_are_deterministic_and_ignore_future_fields() -> None:
    state = sample_state()
    state["unknown_future_field"] = {"anything": True}
    assert decide_move(state) == decide_move(deepcopy(state))


def test_phase_one_protects_a_guaranteed_qualifying_finish() -> None:
    state = sample_state()
    state.update(
        {
            "hand_number": 99,
            "your_stack": 230,
            "your_number": 13,
            "community_number": 13,
        }
    )
    state["players"][0]["stack"] = 230
    assert decide_move(state) == {"action": "fold"}


def test_http_routes_and_health() -> None:
    # Avoid consuming the MCP session manager's one-shot lifespan in this REST
    # test; the dedicated MCP test owns the application's lifespan context.
    client = TestClient(app)
    root = client.post("/move", json=sample_state())
    namespaced = client.post("/showdown/move", json=sample_state())
    health = client.get("/showdown/health")

    assert root.status_code == 200
    assert root.json() == {"action": "fold"}
    assert namespaced.json() == root.json()
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
