from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.challenges.showdown import (
    _clear_rule_memory_for_tests,
    _compare_rule,
    _equity_estimate,
    decide_move,
    pre_reveal_equity,
    showdown_equity,
)
from app.main import app


@pytest.fixture(autouse=True)
def clear_rule_memory() -> None:
    _clear_rule_memory_for_tests()


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


RULE_PROBES = (
    (11, 2, 1),
    (12, 5, 4),
    (4, 3, 12),
    (2, 11, 12),
    (9, 2, 10),
    (7, 1, 1),
    (2, 4, 4),
    (9, 10, 1),
    (9, 4, 12),
    (11, 12, 9),
    (7, 4, 8),
    (10, 5, 13),
    (1, 13, 13),
    (3, 12, 7),
    (6, 5, 3),
    (4, 13, 6),
    (2, 6, 6),
    (10, 5, 13),
    (1, 12, 8),
    (9, 2, 7),
)


def phase_two_state(rule_name: str = "obsidian") -> dict[str, Any]:
    state = sample_state()
    state.update(
        {
            "match_id": "phase2-attempt1-leg1",
            "phase": 2,
            "table_rule": rule_name,
            "leg_number": 1,
            "total_legs": 4,
            "hand_number": 1,
            "total_hands": 40,
            "round": "pre_reveal",
            "community_number": None,
            "your_number": 7,
            "pot": 3,
            "to_call": 1,
            "min_raise_to": 4,
            "max_raise_to": 200,
            "legal_actions": ["fold", "call", "raise"],
            "current_hand_actions": [],
            "recent_hands": [],
            "your_stack": 199,
        }
    )
    state["players"][0].update({"bet_this_round": 1, "stack": 199})
    state["players"][1].update({"bet_this_round": 2, "stack": 198})
    return state


def rule_history(candidate: str) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for hand_number, (first, second, community) in enumerate(RULE_PROBES, 1):
        result = _compare_rule(candidate, first, second, community)
        winners = [0] if result > 0 else [1] if result < 0 else [0, 1]
        histories.append(
            {
                "hand_number": hand_number,
                "community_number": community,
                "winners": winners,
                "pot": 4,
                "shown_numbers": {"0": first, "1": second},
                "actions": [],
            }
        )
    return histories


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


def test_phase_two_explores_unknown_rule_without_risking_a_raise() -> None:
    state = phase_two_state()
    assert decide_move(state) == {"action": "call"}

    state.update(
        {
            "round": "post_reveal",
            "community_number": 6,
            "pot": 6,
            "to_call": 2,
            "min_raise_to": 4,
            "max_raise_to": 198,
            "legal_actions": ["fold", "call", "raise"],
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 2}
            ],
        }
    )
    assert decide_move(state) == {"action": "call"}


def test_phase_two_calls_small_reraise_instead_of_deadlocking_rule_learning() -> None:
    state = phase_two_state("leg-two-rule")
    state.update(
        {
            "your_number": 13,
            "your_stack": 197,
            "pot": 7,
            "to_call": 3,
            "min_raise_to": 8,
            "max_raise_to": 199,
            "legal_actions": ["fold", "call", "raise"],
            "current_hand_actions": [
                {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 2},
                {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 5},
            ],
        }
    )
    state["players"][0].update({"bet_this_round": 2, "stack": 197})
    state["players"][1].update({"bet_this_round": 5, "stack": 195})

    assert decide_move(state) == {"action": "call"}

    state.update(
        {
            "round": "post_reveal",
            "community_number": 6,
            "your_stack": 194,
            "pot": 15,
            "to_call": 5,
            "min_raise_to": 10,
            "max_raise_to": 194,
            "current_hand_actions": [
                *state["current_hand_actions"],
                {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 5},
                {"round": "post_reveal", "seat": 0, "action": "check"},
                {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 5},
            ],
        }
    )
    state["players"][0].update({"bet_this_round": 0, "stack": 194})
    state["players"][1].update({"bet_this_round": 5, "stack": 190})

    assert decide_move(state) == {"action": "call"}


def test_phase_two_oversized_minimum_raise_falls_back_to_call() -> None:
    state = phase_two_state("known-standard")
    state.update(
        {
            "hand_number": 21,
            "round": "post_reveal",
            "your_number": 7,
            "community_number": 7,
            "your_stack": 180,
            "pot": 40,
            "to_call": 20,
            "min_raise_to": 150,
            "max_raise_to": 180,
            "legal_actions": ["fold", "call", "raise"],
            "recent_hands": rule_history("standard"),
        }
    )
    state["players"][0]["bet_this_round"] = 0
    state["players"][1]["bet_this_round"] = 20

    assert decide_move(state) == {"action": "call"}


def test_phase_two_learns_rule_from_showdowns_and_remembers_codename() -> None:
    state = phase_two_state("persistent-onyx")
    state["recent_hands"] = rule_history("low_card")
    estimate = _equity_estimate(state, 1, 7)

    assert estimate.model == "low_card"
    assert estimate.confidence > 0.90
    assert estimate.equity > 0.90

    retry = phase_two_state("persistent-onyx")
    retry["match_id"] = "phase2-attempt2-leg1"
    remembered = _equity_estimate(retry, 13, 7)
    assert remembered.model == "low_card"
    assert remembered.observations == len(RULE_PROBES)
    assert remembered.equity < 0.10

    unrelated = _equity_estimate(phase_two_state("different-rule"), 13, 7)
    assert unrelated.observations == 0
    assert unrelated.confidence < 0.10


@pytest.mark.parametrize(
    "candidate",
    [
        "standard",
        "pair_low",
        "high_card",
        "low_card",
        "pair_bad_high",
        "pair_bad_low",
        "center_high",
        "extreme_low",
        "community_switch",
        "community_switch_inverse",
        "clockwise",
        "counterclockwise",
    ],
)
def test_phase_two_identifies_representative_hidden_rule_families(
    candidate: str,
) -> None:
    state = phase_two_state(f"rule-{candidate}")
    state["recent_hands"] = rule_history(candidate)
    estimate = _equity_estimate(state, 8, 5)
    assert estimate.model == candidate
    assert estimate.confidence > 0.90


def test_phase_two_uses_learned_rule_for_value_betting() -> None:
    state = phase_two_state("reverse-ranking")
    state.update(
        {
            "hand_number": 21,
            "round": "post_reveal",
            "your_number": 1,
            "community_number": 7,
            "pot": 8,
            "to_call": 0,
            "min_raise_to": 2,
            "max_raise_to": 199,
            "legal_actions": ["check", "bet"],
            "current_hand_actions": [],
            "recent_hands": rule_history("low_card"),
        }
    )
    result = decide_move(state)
    assert result["action"] == "bet"
    assert state["min_raise_to"] <= result["amount"] <= state["max_raise_to"]


def test_phase_two_protects_a_guaranteed_plus_twenty_five_leg() -> None:
    state = phase_two_state()
    state.update(
        {
            "hand_number": 39,
            "your_stack": 240,
            "round": "post_reveal",
            "your_number": 7,
            "community_number": 7,
            "pot": 20,
            "to_call": 8,
            "min_raise_to": 16,
            "max_raise_to": 240,
            "legal_actions": ["fold", "call", "raise"],
        }
    )
    state["players"][0]["stack"] = 240
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
