from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from time import perf_counter
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.challenges.showdown as showdown_module
from app.challenges.showdown import (
    FEATURE_MODEL_NAME,
    _clear_rule_memory_for_tests,
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


def phase_three_state(rule_name: str = "obsidian") -> dict[str, Any]:
    state = phase_two_state(rule_name)
    state.update(
        {
            "match_id": "phase3-attempt1-leg1",
            "phase": 3,
            "hand_number": 1,
            "total_hands": 60,
            "your_seat": 0,
            "button_seat": 3,
            "pot": 3,
            "to_call": 2,
        }
    )
    names = ["you", "Dana", "Miles", "Theo", "Rhea", "Bram"]
    state["players"] = [
        {
            "seat": seat,
            "name": name,
            "folded": False,
            "chip_delta": 0,
            "bet_this_round": 0,
            "stack": 200,
            "all_in": False,
            "busted": False,
        }
        for seat, name in enumerate(names)
    ]
    state["players"][0].update({"stack": 200, "bet_this_round": 0})
    state["players"][4].update({"stack": 199, "bet_this_round": 1})
    state["players"][5].update({"stack": 198, "bet_this_round": 2})
    state["your_stack"] = 200
    return state


HiddenRank = Callable[[int, int], tuple[int | bool, ...]]


def hidden_result(
    rank: HiddenRank,
    first: int,
    second: int,
    community: int,
) -> int:
    first_rank = rank(first, community)
    second_rank = rank(second, community)
    return (first_rank > second_rank) - (first_rank < second_rank)


def hidden_history(rank: HiddenRank) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for hand_number, (first, second, community) in enumerate(RULE_PROBES, 1):
        result = hidden_result(rank, first, second, community)
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


def hidden_multiway_history(rank: HiddenRank) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for hand_number in range(1, 13):
        community = (hand_number * 5) % 13 + 1
        numbers = {
            seat: ((hand_number * (seat + 2) + seat * 3) % 13) + 1 for seat in range(6)
        }
        best = max(rank(number, community) for number in numbers.values())
        winners = [
            seat for seat, number in numbers.items() if rank(number, community) == best
        ]
        histories.append(
            {
                "hand_number": hand_number,
                "community_number": community,
                "winners": winners,
                "pot": 18,
                "shown_numbers": {
                    str(seat): number for seat, number in numbers.items()
                },
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


def test_phase_two_pays_a_moderate_early_information_premium() -> None:
    state = phase_two_state("unseen-rule")
    state.update(
        {
            "your_stack": 190,
            "pot": 10,
            "to_call": 10,
            "legal_actions": ["fold", "call", "raise"],
            "current_hand_actions": [
                {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 12}
            ],
        }
    )

    assert decide_move(state) == {"action": "call"}


def test_phase_two_oversized_minimum_raise_falls_back_to_call() -> None:
    state = phase_two_state("known-standard")
    state.update(
        {
            "hand_number": 21,
            "round": "post_reveal",
            "your_number": 10,
            "community_number": 7,
            "your_stack": 180,
            "pot": 40,
            "to_call": 20,
            "min_raise_to": 150,
            "max_raise_to": 180,
            "legal_actions": ["fold", "call", "raise"],
            "recent_hands": hidden_history(lambda number, _community: (number,)),
        }
    )
    state["players"][0]["bet_this_round"] = 0
    state["players"][1]["bet_this_round"] = 20

    assert decide_move(state) == {"action": "call"}


def test_phase_two_learns_rule_from_showdowns_and_remembers_codename() -> None:
    state = phase_two_state("persistent-onyx")
    state["recent_hands"] = hidden_history(lambda number, _community: (-number,))
    estimate = _equity_estimate(state, 1, 7)

    assert estimate.model == FEATURE_MODEL_NAME
    assert estimate.confidence > 0.70
    assert estimate.equity > 0.80

    retry = phase_two_state("persistent-onyx")
    retry["match_id"] = "phase2-attempt2-leg1"
    remembered = _equity_estimate(retry, 13, 7)
    assert remembered.observations == len(RULE_PROBES)
    assert remembered.equity < 0.40

    unrelated = _equity_estimate(phase_two_state("different-rule"), 13, 7)
    assert unrelated.observations == 0
    assert unrelated.confidence == 0.0
    assert unrelated.equity == 0.5


def test_phase_two_uses_the_required_exact_hypothesis_catalogue() -> None:
    required = {
        "standard",
        "low",
        "proximity",
        "anti-proximity",
        "clockwise",
        "counter-clockwise",
        "odd-high",
        "prime-high",
        "target-14-under",
        "target-21-nearest",
    }
    assert required <= set(showdown_module.RULE_HYPOTHESES)


def test_exact_hypothesis_locks_only_one_fully_consistent_rule() -> None:
    state = phase_two_state("exact-lowball")
    state["recent_hands"] = hidden_history(showdown_module.RULE_HYPOTHESES["low"])
    estimate = _equity_estimate(state, 1, 7)

    assert estimate.locked_rule == "low"
    assert estimate.confidence == 1.0
    assert estimate.percentile > 0.90


def test_non_catalogue_rule_uses_pairwise_tournament_fallback() -> None:
    rank = lambda number, community: (number % 3, -abs(number - community))
    state = phase_two_state("non-catalogue")
    state["recent_hands"] = hidden_history(rank)
    estimate = _equity_estimate(state, 7, 7)
    observations = showdown_module._rule_observations("non-catalogue")
    model = showdown_module._learn_rule_model("non-catalogue", observations)

    assert estimate.locked_rule is None
    assert model.candidates == ()
    first = observations[0]
    assert (
        model.relations[first.community - 1][first.first_number - 1][
            first.second_number - 1
        ]
        == first.result
    )


@pytest.mark.parametrize(
    ("label", "rank"),
    [
        ("standard", lambda number, community: (number == community, number)),
        ("reverse-pairs", lambda number, community: (number == community, -number)),
        ("high", lambda number, _community: (number,)),
        ("low", lambda number, _community: (-number,)),
        ("odd-first", lambda number, _community: (number % 2, number)),
        (
            "prime-first",
            lambda number, _community: (number in {2, 3, 5, 7, 11, 13}, number),
        ),
        (
            "near-community",
            lambda number, community: (-abs(number - community), number),
        ),
        ("far-community", lambda number, community: (abs(number - community), number)),
        ("near-seven", lambda number, _community: (-abs(number - 7), number)),
        (
            "community-reversal",
            lambda number, community: (number if community <= 7 else -number,),
        ),
        ("cyclic", lambda number, community: ((number - community) % 13,)),
    ],
)
def test_phase_two_generalizes_to_unseen_card_combinations(
    label: str,
    rank: HiddenRank,
) -> None:
    state = phase_two_state(f"rule-{label}")
    state["recent_hands"] = hidden_history(rank)

    absolute_error = 0.0
    for community in range(1, 14):
        for number in range(1, 14):
            estimate = _equity_estimate(state, number, community)
            true_equity = (
                sum(
                    1.0
                    if hidden_result(rank, number, opponent, community) > 0
                    else 0.5
                    if hidden_result(rank, number, opponent, community) == 0
                    else 0.0
                    for opponent in range(1, 14)
                )
                / 13
            )
            absolute_error += abs(estimate.equity - true_equity)

    assert estimate.model == FEATURE_MODEL_NAME
    assert estimate.confidence > 0.65
    assert absolute_error / (13 * 13) < 0.22


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
            "recent_hands": hidden_history(lambda number, _community: (-number,)),
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


def test_phase_three_unknown_rule_gathers_information_at_bounded_cost() -> None:
    state = phase_three_state("new-multiway-rule")
    assert decide_move(state) == {"action": "call"}

    state.update(
        {
            "round": "post_reveal",
            "community_number": 8,
            "pot": 24,
            "to_call": 7,
            "min_raise_to": 14,
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 4, "action": "bet", "amount": 7}
            ],
        }
    )
    assert decide_move(state) == {"action": "call"}


def test_phase_three_learns_many_valid_comparisons_per_showdown() -> None:
    state = phase_three_state("multiway-low")
    state["recent_hands"] = hidden_multiway_history(
        lambda number, _community: (-number,)
    )
    low = _equity_estimate(state, 1, 7)
    high = _equity_estimate(state, 13, 7)

    assert low.opponents == 5
    assert low.observations > len(state["recent_hands"])
    assert low.confidence > 0.70
    assert low.equity > high.equity


def test_phase_three_equity_accounts_for_every_live_opponent() -> None:
    state = phase_three_state("multiway-high")
    state["recent_hands"] = hidden_multiway_history(
        lambda number, _community: (number,)
    )
    six_seat_equity = _equity_estimate(state, 12, 7)

    for player in state["players"][2:]:
        player["folded"] = True
    heads_up_equity = _equity_estimate(state, 12, 7)

    assert six_seat_equity.opponents == 5
    assert heads_up_equity.opponents == 1
    assert six_seat_equity.equity < heads_up_equity.equity


def test_phase_three_unknown_rule_uses_a_conservative_hypothesis_ensemble() -> None:
    state = phase_three_state("never-seen")
    state.update({"round": "post_reveal", "community_number": 7})

    estimate = _equity_estimate(state, 7, 7)

    assert estimate.observations == 0
    assert estimate.robust_equity <= estimate.equity
    assert estimate.robust_percentile <= estimate.percentile
    assert 0.0 <= estimate.premium_support <= 1.0


def test_phase_three_filters_folded_and_busted_seats_but_keeps_all_ins() -> None:
    state = phase_three_state("live-seat-filter")
    state["players"][1]["folded"] = True
    state["players"][2]["busted"] = True
    state["players"][3]["all_in"] = True
    estimate = _equity_estimate(state, 7, 7)

    assert estimate.opponents == 3
    assert 0.0 < estimate.equity < 1.0
    assert estimate.robust_equity <= estimate.equity


def test_phase_three_uses_official_six_seat_action_order() -> None:
    state = phase_three_state("position-rule")
    state["button_seat"] = 0

    state["your_seat"] = 0
    assert not showdown_module._is_late_position(state, "pre_reveal")
    assert showdown_module._is_late_position(state, "post_reveal")

    state["your_seat"] = 2
    assert showdown_module._is_late_position(state, "pre_reveal")
    assert not showdown_module._is_late_position(state, "post_reveal")


def test_phase_three_only_a_strict_chip_majority_is_an_early_hard_lock() -> None:
    state = phase_three_state("majority-high")
    state.update(
        {
            "hand_number": 20,
            "round": "post_reveal",
            "your_number": 13,
            "community_number": 6,
            "your_stack": 650,
            "pot": 50,
            "to_call": 25,
            "legal_actions": ["fold", "call", "raise"],
            "min_raise_to": 50,
            "max_raise_to": 650,
            "recent_hands": hidden_multiway_history(
                lambda number, _community: (number,)
            ),
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 25}
            ],
        }
    )
    state["players"][0].update({"stack": 650, "chip_delta": 450})
    for player, delta in zip(state["players"][1:], [100, -50, -100, -200, -200]):
        player["chip_delta"] = delta

    assert decide_move(state) == {"action": "fold"}


def test_phase_three_does_not_false_lock_an_ordinary_early_lead() -> None:
    state = phase_three_state("ordinary-lead-high")
    state.update(
        {
            "hand_number": 18,
            "round": "post_reveal",
            "your_number": 13,
            "community_number": 6,
            "your_stack": 230,
            "pot": 40,
            "to_call": 15,
            "legal_actions": ["fold", "call", "raise"],
            "min_raise_to": 30,
            "max_raise_to": 230,
            "recent_hands": hidden_multiway_history(
                lambda number, _community: (number,)
            ),
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 15}
            ],
        }
    )
    state["players"][0].update({"stack": 230, "chip_delta": 30})
    for player, delta in zip(state["players"][1:], [10, 5, -5, -15, -25]):
        player["chip_delta"] = delta

    assert decide_move(state)["action"] == "raise"


def test_phase_three_intercepts_an_exposed_early_chip_leader_with_the_nuts() -> None:
    state = phase_three_state("leader-high")
    state.update(
        {
            "hand_number": 15,
            "round": "post_reveal",
            "your_number": 13,
            "community_number": 6,
            "your_stack": 200,
            "pot": 80,
            "to_call": 30,
            "legal_actions": ["fold", "call", "raise"],
            "min_raise_to": 60,
            "max_raise_to": 200,
            "recent_hands": hidden_multiway_history(
                lambda number, _community: (number,)
            ),
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 30}
            ],
        }
    )
    for player, delta in zip(state["players"], [0, 100, -10, -20, -30, -40]):
        player["chip_delta"] = delta

    assert decide_move(state) == {"action": "raise", "amount": 200}


def test_phase_three_checks_a_merely_average_hand_into_five_ranges() -> None:
    state = phase_three_state("multiway-value-high")
    state.update(
        {
            "hand_number": 18,
            "round": "post_reveal",
            "your_number": 9,
            "community_number": 6,
            "pot": 24,
            "to_call": 0,
            "legal_actions": ["check", "bet"],
            "min_raise_to": 2,
            "max_raise_to": 200,
            "recent_hands": hidden_multiway_history(
                lambda number, _community: (number,)
            ),
            "current_hand_actions": [],
        }
    )

    assert decide_move(state) == {"action": "check"}


def test_phase_three_respects_tag_range_but_calls_a_maniac_heads_up() -> None:
    def facing(name_seat: int) -> dict[str, Any]:
        state = phase_three_state("range-high")
        state.update(
            {
                "hand_number": 24,
                "round": "post_reveal",
                "your_number": 11,
                "community_number": 6,
                "your_stack": 200,
                "pot": 120,
                "to_call": 80,
                "legal_actions": ["fold", "call"],
                "min_raise_to": None,
                "max_raise_to": None,
                "recent_hands": hidden_multiway_history(
                    lambda number, _community: (number,)
                ),
                "current_hand_actions": [
                    {
                        "round": "post_reveal",
                        "seat": name_seat,
                        "action": "bet",
                        "amount": 80,
                    }
                ],
            }
        )
        for player in state["players"][1:]:
            player["folded"] = player["seat"] != name_seat
        return state

    assert decide_move(facing(1)) == {"action": "fold"}  # Dana: TAG prior
    assert decide_move(facing(5)) == {"action": "call"}  # Bram: maniac prior


def test_phase_three_protects_a_qualifying_strict_late_lead() -> None:
    state = phase_three_state("known-high")
    state.update(
        {
            "hand_number": 58,
            "round": "post_reveal",
            "community_number": 7,
            "your_number": 1,
            "pot": 35,
            "to_call": 12,
            "legal_actions": ["fold", "call", "raise"],
            "min_raise_to": 24,
            "recent_hands": hidden_multiway_history(
                lambda number, _community: (number,)
            ),
        }
    )
    deltas = [26, 10, 4, -2, -15, -23]
    for player, delta in zip(state["players"], deltas, strict=True):
        player["chip_delta"] = delta

    assert decide_move(state) == {"action": "fold"}


def test_phase_three_persists_individual_opponent_metrics_across_legs() -> None:
    state = phase_three_state("profile-rule")
    state["recent_hands"] = [
        {
            "hand_number": hand,
            "community_number": 7,
            "winners": [3],
            "shown_numbers": {"3": 13, "0": 2},
            "actions": [
                {"round": "pre_reveal", "seat": 3, "action": "raise", "amount": 8},
                {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 8},
                {"round": "post_reveal", "seat": 3, "action": "bet", "amount": 16},
                {"round": "post_reveal", "seat": 0, "action": "fold"},
            ],
        }
        for hand in range(1, 11)
    ]
    state.update(
        {
            "round": "post_reveal",
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 3, "action": "bet", "amount": 20}
            ],
        }
    )
    first = showdown_module._opponent_profile(state)

    retry = phase_three_state("profile-rule")
    retry.update(
        {
            "match_id": "phase3-attempt2-leg2",
            "leg_number": 2,
            "round": "post_reveal",
            "recent_hands": [],
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 3, "action": "bet", "amount": 20}
            ],
        }
    )
    remembered = showdown_module._opponent_profile(retry)

    assert first.name == remembered.name == "Theo"
    assert remembered.observations == 10
    assert remembered.vpip > 0.75
    assert remembered.pfr > 0.65
    assert remembered.aggression_factor > 5
    assert remembered.archetype == "maniac"


def test_phase_three_reshoves_premium_hand_against_early_maniac() -> None:
    state = phase_three_state("early-standard")
    state.update(
        {
            "hand_number": 12,
            "round": "post_reveal",
            "your_number": 7,
            "community_number": 7,
            "pot": 170,
            "to_call": 150,
            "your_stack": 200,
            "legal_actions": ["fold", "call", "raise"],
            "min_raise_to": 200,
            "max_raise_to": 200,
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 3, "action": "bet", "amount": 150}
            ],
            "recent_hands": hidden_multiway_history(
                lambda number, community: (number == community, number)
            ),
        }
    )

    assert decide_move(state) == {"action": "raise", "amount": 200}


def test_phase_three_midgame_steals_from_late_position() -> None:
    state = phase_three_state("mid-high")
    state.update(
        {
            "hand_number": 30,
            # With button 4, seat 0 is the big blind and acts last before reveal.
            "button_seat": 4,
            "round": "pre_reveal",
            "your_number": 7,
            "community_number": None,
            "pot": 3,
            "to_call": 0,
            "legal_actions": ["check", "bet"],
            "min_raise_to": 2,
            "max_raise_to": 200,
            "recent_hands": hidden_multiway_history(
                lambda number, _community: (number,)
            ),
        }
    )
    for player in state["players"][1:4]:
        player["folded"] = True
    result = decide_move(state)

    assert result["action"] == "bet"
    assert 2 <= result["amount"] <= 200


def test_phase_three_trailing_endgame_shoves_top_quartile_at_leader() -> None:
    state = phase_three_state("end-high")
    state.update(
        {
            "hand_number": 57,
            "round": "post_reveal",
            "your_number": 13,
            "community_number": 6,
            "pot": 42,
            "to_call": 20,
            "your_stack": 180,
            "legal_actions": ["fold", "call", "raise"],
            "min_raise_to": 40,
            "max_raise_to": 180,
            "current_hand_actions": [
                {"round": "post_reveal", "seat": 3, "action": "bet", "amount": 20}
            ],
            "recent_hands": hidden_multiway_history(
                lambda number, _community: (number,)
            ),
        }
    )
    deltas = [0, 5, -4, 22, -8, -15]
    for player, delta in zip(state["players"], deltas, strict=True):
        player["chip_delta"] = delta

    assert decide_move(state) == {"action": "raise", "amount": 180}


@pytest.mark.parametrize(
    ("legal", "to_call", "minimum", "maximum"),
    [
        (["check", "bet"], 0, 3, 80),
        (["fold", "call", "raise"], 9, 18, 100),
        (["check"], 0, None, None),
        (["fold", "call"], 25, None, None),
    ],
)
def test_phase_three_always_obeys_the_authoritative_action_contract(
    legal: list[str],
    to_call: int,
    minimum: int | None,
    maximum: int | None,
) -> None:
    state = phase_three_state("contract-rule")
    state["recent_hands"] = hidden_multiway_history(
        lambda number, community: (number == community, number)
    )
    state.update(
        {
            "hand_number": 30,
            "round": "post_reveal",
            "community_number": 6,
            "pot": 45,
            "to_call": to_call,
            "legal_actions": legal,
            "min_raise_to": minimum,
            "max_raise_to": maximum,
        }
    )
    for number in range(1, 14):
        state["your_number"] = number
        result = decide_move(state)
        assert result["action"] in legal
        if result["action"] in {"bet", "raise"}:
            assert minimum is not None and maximum is not None
            assert minimum <= result["amount"] <= maximum
        else:
            assert set(result) == {"action"}


def test_phase_three_first_uncached_decision_is_under_fifty_milliseconds() -> None:
    state = phase_three_state("latency-exact")
    state.update(
        {
            "hand_number": 30,
            "round": "post_reveal",
            "community_number": 7,
            "your_number": 13,
            "pot": 50,
            "to_call": 8,
            "legal_actions": ["fold", "call", "raise"],
            "min_raise_to": 16,
            "max_raise_to": 200,
            "recent_hands": hidden_multiway_history(
                lambda number, community: (number == community, number)
            ),
        }
    )

    started = perf_counter()
    result = decide_move(state)
    elapsed = perf_counter() - started

    assert result["action"] in state["legal_actions"]
    assert elapsed < 0.050


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
