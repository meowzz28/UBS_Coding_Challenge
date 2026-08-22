"""SHOWDOWN Phase 1 and Phase 2 betting strategy and HTTP routes."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, TypedDict, cast

from fastapi import APIRouter

Action = Literal["check", "call", "bet", "raise", "fold"]
AGGRESSIVE_ACTIONS = {"bet", "raise"}
RULE_CANDIDATES = (
    "standard",
    "pair_low",
    "high_card",
    "low_card",
    "pair_bad_high",
    "pair_bad_low",
    "closest_split",
    "closest_high",
    "closest_low",
    "farthest_split",
    "farthest_high",
    "farthest_low",
    "center_split",
    "center_high",
    "center_low",
    "extreme_split",
    "extreme_high",
    "extreme_low",
    "community_switch",
    "community_switch_inverse",
    "clockwise",
    "counterclockwise",
)


class Move(TypedDict, total=False):
    action: Action
    amount: int


@dataclass(frozen=True)
class OpponentProfile:
    """Smoothed tendencies reconstructed from the rolling public history."""

    fold_to_aggression: float
    aggression_rate: float
    observations: int


@dataclass(frozen=True)
class ShowdownObservation:
    first_number: int
    second_number: int
    community: int
    result: int


@dataclass(frozen=True)
class EquityEstimate:
    equity: float
    confidence: float
    observations: int
    model: str


_RULE_MEMORY: dict[
    str, dict[tuple[str, int, int, int, int, int], ShowdownObservation]
] = {}
_RULE_MEMORY_LOCK = Lock()


router = APIRouter(tags=["showdown"])


@router.get("/showdown/health")
def challenge_health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/move", response_model=None)
@router.post("/showdown/move", response_model=None)
def move(payload: dict[str, Any]) -> Move:
    """Return one fast, deterministic, legal move for a SHOWDOWN turn."""

    return decide_move(payload)


def decide_move(state: Mapping[str, Any]) -> Move:
    """Choose a move using learned rules, equity, price, and opponent history."""

    legal = _legal_actions(state.get("legal_actions"))
    if not legal:
        # Valid coordinator requests always contain this field. Keeping a benign
        # response here makes malformed probes cheap and side-effect-free.
        return {"action": "check"}

    number = _bounded_int(state.get("your_number"), 1, 13, default=7)
    community = _optional_bounded_int(state.get("community_number"), 1, 13)
    round_name = str(state.get("round", "pre_reveal"))
    to_call = max(0, _int(state.get("to_call"), 0))
    pot = max(0, _int(state.get("pot"), 0))
    stack = max(0, _int(state.get("your_stack"), 0))
    own_bet = _own_round_bet(state)
    profile = _opponent_profile(state)
    equity = _equity_estimate(state, number, community)

    # Both phases are threshold-scored. Once the live stack can absorb every
    # remaining forced bet and retain the qualifying delta, remove variance.
    if _phase_target_locked(state, stack):
        if to_call > 0:
            return _first_legal(legal, "fold", "call", "check")
        return _first_legal(legal, "check", "call", "fold")

    # At the start of an opaque Phase 2 leg, reach inexpensive showdowns to learn
    # its rule instead of risking chips based on the standard table assumption.
    if (
        _int(state.get("phase"), 0) == 2
        and equity.observations < 8
        and (equity.observations < 6 or equity.confidence < 0.72)
    ):
        return _learning_move(
            state,
            legal,
            to_call,
            pot,
            stack,
            equity.equity,
        )

    if round_name == "post_reveal" and community is not None:
        return _post_reveal_move(
            state=state,
            legal=legal,
            to_call=to_call,
            pot=pot,
            stack=stack,
            own_bet=own_bet,
            profile=profile,
            equity=equity.equity,
        )

    return _pre_reveal_move(
        state=state,
        legal=legal,
        to_call=to_call,
        pot=pot,
        stack=stack,
        own_bet=own_bet,
        profile=profile,
        equity=equity.equity,
    )


def showdown_equity(your_number: int, community_number: int) -> float:
    """Exact heads-up showdown equity against a uniformly dealt number."""

    if your_number == community_number:
        # Twelve outright wins and one split against the same pair.
        return 12.5 / 13
    if community_number < your_number:
        # The community number is removed from the lower winning numbers because
        # it makes the opponent a pair; an equal private number splits.
        return (your_number - 1.5) / 13
    return (your_number - 0.5) / 13


def pre_reveal_equity(your_number: int) -> float:
    """Exact equity averaged over all 13 opponent and community numbers."""

    # Enumeration simplifies to a linear sequence from 18.5/169 to 150.5/169.
    return (11 * your_number + 7.5) / 169


def _equity_estimate(
    state: Mapping[str, Any], your_number: int, community: int | None
) -> EquityEstimate:
    phase = _int(state.get("phase"), 1)
    if phase == 1:
        equity = (
            showdown_equity(your_number, community)
            if community is not None
            else pre_reveal_equity(your_number)
        )
        return EquityEstimate(equity, 1.0, 100, "standard")

    rule_name = str(state.get("table_rule", ""))
    _remember_showdowns(state, rule_name)
    observations = _rule_observations(rule_name)
    weights = _candidate_weights(observations)
    model = RULE_CANDIDATES[max(range(len(weights)), key=weights.__getitem__)]
    confidence = max(weights)

    if community is None:
        total = 0.0
        for candidate, weight in zip(RULE_CANDIDATES, weights, strict=True):
            candidate_total = (
                sum(
                    _candidate_equity(candidate, your_number, revealed)
                    for revealed in range(1, 14)
                )
                / 13
            )
            total += weight * candidate_total
    else:
        total = sum(
            weight * _candidate_equity(candidate, your_number, community)
            for candidate, weight in zip(RULE_CANDIDATES, weights, strict=True)
        )
    return EquityEstimate(total, confidence, len(observations), model)


def _remember_showdowns(state: Mapping[str, Any], rule_name: str) -> None:
    histories = state.get("recent_hands")
    if (
        not rule_name
        or not isinstance(histories, Sequence)
        or isinstance(histories, (str, bytes))
    ):
        return

    match_id = str(state.get("match_id", ""))
    leg_number = _int(state.get("leg_number"), 0)
    additions: dict[tuple[str, int, int, int, int, int], ShowdownObservation] = {}
    for hand in histories:
        if not isinstance(hand, Mapping):
            continue
        community = _optional_bounded_int(hand.get("community_number"), 1, 13)
        shown = hand.get("shown_numbers")
        winners = hand.get("winners")
        hand_number = _int(hand.get("hand_number"), 0)
        if (
            community is None
            or not isinstance(shown, Mapping)
            or not isinstance(winners, Sequence)
            or isinstance(winners, (str, bytes))
            or hand_number < 1
        ):
            continue

        seats: list[tuple[int, int]] = []
        for raw_seat, raw_number in shown.items():
            try:
                seat = int(raw_seat)
            except (TypeError, ValueError):
                continue
            number = _optional_bounded_int(raw_number, 1, 13)
            if number is not None:
                seats.append((seat, number))
        if len(seats) != 2:
            continue
        seats.sort()
        winner_seats = {
            winner
            for winner in winners
            if isinstance(winner, int) and not isinstance(winner, bool)
        }
        if winner_seats == {seats[0][0], seats[1][0]}:
            result = 0
        elif winner_seats == {seats[0][0]}:
            result = 1
        elif winner_seats == {seats[1][0]}:
            result = -1
        else:
            continue

        key = (
            match_id,
            leg_number,
            hand_number,
            seats[0][1],
            seats[1][1],
            community,
        )
        additions[key] = ShowdownObservation(
            seats[0][1], seats[1][1], community, result
        )

    if additions:
        with _RULE_MEMORY_LOCK:
            memory = _RULE_MEMORY.setdefault(rule_name, {})
            memory.update(additions)
            while len(memory) > 500:
                memory.pop(next(iter(memory)))


def _rule_observations(rule_name: str) -> tuple[ShowdownObservation, ...]:
    with _RULE_MEMORY_LOCK:
        return tuple(_RULE_MEMORY.get(rule_name, {}).values())


def _candidate_weights(
    observations: Sequence[ShowdownObservation],
) -> tuple[float, ...]:
    if not observations:
        uniform = 1 / len(RULE_CANDIDATES)
        return (uniform,) * len(RULE_CANDIDATES)

    scores: list[float] = []
    for candidate in RULE_CANDIDATES:
        errors = sum(
            _compare_rule(
                candidate,
                observation.first_number,
                observation.second_number,
                observation.community,
            )
            != observation.result
            for observation in observations
        )
        scores.append(-3.5 * errors)
    peak = max(scores)
    raw = [math.exp(score - peak) for score in scores]
    total = sum(raw)
    return tuple(weight / total for weight in raw)


def _candidate_equity(candidate: str, your_number: int, community: int) -> float:
    points = 0.0
    for opponent_number in range(1, 14):
        result = _compare_rule(candidate, your_number, opponent_number, community)
        points += 1.0 if result > 0 else 0.5 if result == 0 else 0.0
    return points / 13


def _compare_rule(candidate: str, first: int, second: int, community: int) -> int:
    first_rank = _rule_rank(candidate, first, community)
    second_rank = _rule_rank(candidate, second, community)
    return (first_rank > second_rank) - (first_rank < second_rank)


def _rule_rank(candidate: str, number: int, community: int) -> tuple[int, ...]:
    pair = int(number == community)
    distance = abs(number - community)
    center_distance = abs(number - 7)
    if candidate == "standard":
        return (pair, number)
    if candidate == "pair_low":
        return (pair, -number)
    if candidate == "high_card":
        return (number,)
    if candidate == "low_card":
        return (-number,)
    if candidate == "pair_bad_high":
        return (-pair, number)
    if candidate == "pair_bad_low":
        return (-pair, -number)
    if candidate.startswith("closest"):
        return _distance_rank(candidate, -distance, number)
    if candidate.startswith("farthest"):
        return _distance_rank(candidate, distance, number)
    if candidate.startswith("center"):
        return _distance_rank(candidate, -center_distance, number)
    if candidate.startswith("extreme"):
        return _distance_rank(candidate, center_distance, number)
    if candidate == "community_switch":
        return (number if community <= 7 else -number,)
    if candidate == "community_switch_inverse":
        return (-number if community <= 7 else number,)
    if candidate == "clockwise":
        return ((number - community) % 13,)
    if candidate == "counterclockwise":
        return ((community - number) % 13,)
    raise ValueError(f"unknown rule candidate: {candidate}")


def _distance_rank(candidate: str, primary: int, number: int) -> tuple[int, ...]:
    if candidate.endswith("_high"):
        return (primary, number)
    if candidate.endswith("_low"):
        return (primary, -number)
    return (primary,)


def _learning_move(
    state: Mapping[str, Any],
    legal: set[Action],
    to_call: int,
    pot: int,
    stack: int,
    equity: float,
) -> Move:
    if to_call == 0:
        return _first_legal(legal, "check", "call", "fold")

    round_name = str(state.get("round", "pre_reveal"))
    actions = _actions(state.get("current_hand_actions"))
    small_blind = max(1, _int(state.get("small_blind"), 1))
    blind_completion = (
        round_name == "pre_reveal" and not actions and to_call <= small_blind
    )
    risk = to_call / max(1, stack)
    price = _pot_odds(to_call, pot)

    # A passive learning policy used to fold every time the opponent raised the
    # completed blind. That produced no shown numbers, so the opaque rule could
    # never be learned. Pay bounded prices through both rounds until enough
    # labelled showdowns have accumulated; checking remains preferred when free.
    if round_name == "pre_reveal":
        learning_cap = max(4, min(round(pot * 0.85), round(stack * 0.07)))
    else:
        learning_cap = max(5, min(round(pot * 0.90), round(stack * 0.08), 12))
    informative_call = to_call <= learning_cap and risk <= 0.10
    priced_call = risk <= 0.12 and equity >= price - 0.06
    if "call" in legal and (blind_completion or informative_call or priced_call):
        return {"action": "call"}
    return _first_legal(legal, "fold", "call", "check")


def _post_reveal_move(
    *,
    state: Mapping[str, Any],
    legal: set[Action],
    to_call: int,
    pot: int,
    stack: int,
    own_bet: int,
    profile: OpponentProfile,
    equity: float,
) -> Move:
    is_nuts = equity >= 0.93
    facing_bet = to_call > 0

    if facing_bet:
        price = _pot_odds(to_call, pot)
        risk = to_call / max(1, stack)

        # Build the pot with hands that are effectively the nuts under the
        # inferred table rule. Under standard rules this is exactly a pair.
        if is_nuts:
            if _can_aggress(legal):
                fraction = 1.35 if profile.fold_to_aggression < 0.38 else 0.80
                return _aggressive_move(state, legal, own_bet, to_call, pot, fraction)
            return _first_legal(legal, "call", "check", "fold")

        # Betting frequency is evidence about the opponent's range. A normally
        # passive opponent gets more credit; a frequent bettor gets called wider.
        adjusted_equity = _clamp(
            equity + (profile.aggression_rate - 0.34) * 0.20,
            0.0,
            1.0,
        )

        # Raise premium hands for value without turning a marginal edge into an
        # unnecessary tournament-sized pot.
        if adjusted_equity >= 0.78 and risk <= 0.20 and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 0.65)

        call_margin = 0.035 if profile.aggression_rate >= 0.45 else 0.065
        affordable = risk <= 0.22 or (adjusted_equity >= 0.90 and risk <= 0.38)
        if "call" in legal and affordable and adjusted_equity >= price + call_margin:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")

    # When checked to, value bet the strongest portion of the rule-specific range.
    # Occasionally checking the nuts against aggression preserves an inducing line.
    if is_nuts:
        trap = profile.aggression_rate >= 0.48 and _mix(state, "pair-trap") < 0.16
        if not trap and _can_aggress(legal):
            fraction = 0.95 if profile.fold_to_aggression < 0.42 else 0.62
            return _aggressive_move(state, legal, own_bet, 0, pot, fraction)
        return _first_legal(legal, "check", "call", "fold")

    if equity >= 0.79 and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.62)
    if equity >= 0.68 and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.42)

    # Bluff only when the observed fold rate makes the price plausible. The stable
    # hash keeps retries idempotent while still preventing an exploitable fixed line.
    if _can_aggress(legal) and equity <= 0.30:
        bluff_rate = _clamp(
            0.06 + (profile.fold_to_aggression - 0.38) * 0.55,
            0.04,
            0.30,
        )
        if _mix(state, "river-bluff") < bluff_rate:
            return _aggressive_move(state, legal, own_bet, 0, pot, 0.42)

    return _first_legal(legal, "check", "call", "fold")


def _pre_reveal_move(
    *,
    state: Mapping[str, Any],
    legal: set[Action],
    to_call: int,
    pot: int,
    stack: int,
    own_bet: int,
    profile: OpponentProfile,
    equity: float,
) -> Move:
    small_blind = max(1, _int(state.get("small_blind"), 1))
    actions = _actions(state.get("current_hand_actions"))
    blind_completion = to_call <= small_blind and not actions

    # The button's opening decision is a one-chip completion rather than evidence
    # of opponent strength. Continue nearly every hand that is getting the price.
    if blind_completion:
        if equity >= 0.68 and _can_aggress(legal):
            fraction = 0.72 if equity >= 0.79 else 0.52
            return _aggressive_move(state, legal, own_bet, to_call, pot, fraction)
        if equity >= 0.22 and "call" in legal:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")

    if to_call > 0:
        price = _pot_odds(to_call, pot)
        risk = to_call / max(1, stack)
        adjusted_equity = _clamp(
            equity + (profile.aggression_rate - 0.34) * 0.16,
            0.0,
            1.0,
        )

        if risk <= 0.16 and adjusted_equity >= 0.76 and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 0.62)

        # Avoid calling off most of the match with a hand that has not yet seen the
        # community number. Only top rule-specific equity can call a wide shove.
        affordable = risk <= 0.18 or (adjusted_equity >= 0.90 and risk <= 0.30)
        margin = 0.075 if profile.aggression_rate < 0.40 else 0.045
        if "call" in legal and affordable and adjusted_equity >= price + margin:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")

    if equity >= 0.74 and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.62)
    if equity >= 0.62 and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.42)

    if _can_aggress(legal) and equity <= 0.25:
        bluff_rate = _clamp(
            0.04 + (profile.fold_to_aggression - 0.40) * 0.40,
            0.02,
            0.18,
        )
        if _mix(state, "pre-bluff") < bluff_rate:
            return _aggressive_move(state, legal, own_bet, 0, pot, 0.42)

    return _first_legal(legal, "check", "call", "fold")


def _opponent_profile(state: Mapping[str, Any]) -> OpponentProfile:
    your_seat = _int(state.get("your_seat"), -1)
    opponent_actions = 0
    opponent_aggression = 0
    responses = 0
    folds = 0

    histories = state.get("recent_hands")
    if not isinstance(histories, Sequence) or isinstance(histories, (str, bytes)):
        histories = []

    for hand in histories:
        if not isinstance(hand, Mapping):
            continue
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for action in _actions(hand.get("actions")):
            grouped.setdefault(str(action.get("round", "")), []).append(action)

        for round_actions in grouped.values():
            for index, action in enumerate(round_actions):
                seat = _int(action.get("seat"), -2)
                action_name = str(action.get("action", ""))
                if seat != your_seat:
                    opponent_actions += 1
                    if action_name in AGGRESSIVE_ACTIONS:
                        opponent_aggression += 1

                if (
                    seat == your_seat
                    and action_name in AGGRESSIVE_ACTIONS
                    and index + 1 < len(round_actions)
                ):
                    answer = round_actions[index + 1]
                    if _int(answer.get("seat"), -2) != your_seat:
                        responses += 1
                        if str(answer.get("action", "")) == "fold":
                            folds += 1

    # Include this hand for aggression estimation but not fold response—the hand
    # is incomplete and a missing next action is not evidence of a fold.
    for action in _actions(state.get("current_hand_actions")):
        if _int(action.get("seat"), -2) != your_seat:
            opponent_actions += 1
            if str(action.get("action", "")) in AGGRESSIVE_ACTIONS:
                opponent_aggression += 1

    # Conservative beta priors prevent the first few hands from causing extreme
    # over-adjustment: fold rate starts at 1/3 and aggression at 0.30.
    fold_rate = (folds + 2) / (responses + 6)
    aggression_rate = (opponent_aggression + 3) / (opponent_actions + 10)
    return OpponentProfile(fold_rate, aggression_rate, opponent_actions)


def _aggressive_move(
    state: Mapping[str, Any],
    legal: set[Action],
    own_bet: int,
    to_call: int,
    pot: int,
    pot_fraction: float,
) -> Move:
    action: Action
    if "raise" in legal:
        action = "raise"
    elif "bet" in legal:
        action = "bet"
    else:
        return _first_legal(legal, "call", "check", "fold")

    minimum = state.get("min_raise_to")
    maximum = state.get("max_raise_to")
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        return _first_legal(legal - {action}, "call", "check", "fold")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < minimum:
        return _first_legal(legal - {action}, "call", "check", "fold")

    if _int(state.get("phase"), 1) == 2:
        pot_fraction *= 1.20
    desired = own_bet + to_call + max(1, round(pot * pot_fraction))
    if _int(state.get("phase"), 1) == 2:
        # Phase 2 is scored on clearing +25, not on maximizing a lucky stack.
        # Never let a large minimum raise silently turn an ordinary value line
        # into a quarter-match-or-more wager.
        live_stack = max(0, _int(state.get("your_stack"), 0))
        extra_cap = max(4, round(live_stack * 0.24))
        capped_total = own_bet + extra_cap
        if minimum > capped_total:
            return _first_legal(legal - {action}, "call", "check", "fold")
        desired = min(desired, capped_total)
    amount = max(minimum, min(maximum, desired))
    return {"action": action, "amount": amount}


def _legal_actions(value: Any) -> set[Action]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    valid = {"check", "call", "bet", "raise", "fold"}
    return {
        cast(Action, action)
        for action in value
        if isinstance(action, str) and action in valid
    }


def _actions(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _own_round_bet(state: Mapping[str, Any]) -> int:
    your_seat = _int(state.get("your_seat"), -1)
    players = state.get("players")
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        for player in players:
            if (
                isinstance(player, Mapping)
                and _int(player.get("seat"), -2) == your_seat
            ):
                return max(0, _int(player.get("bet_this_round"), 0))
    return 0


def _phase_target_locked(state: Mapping[str, Any], live_stack: int) -> bool:
    phase = _int(state.get("phase"), 0)
    targets = {1: 10, 2: 25}
    if phase not in targets:
        return False

    hand = _int(state.get("hand_number"), 0)
    total = _int(state.get("total_hands"), 0)
    your_seat = _int(state.get("your_seat"), -1)
    button = _int(state.get("button_seat"), -2)
    starting_stack = max(0, _int(state.get("starting_stack"), 200))
    small_blind = max(0, _int(state.get("small_blind"), 1))
    big_blind = max(0, _int(state.get("big_blind"), 2))
    if hand < 1 or total < hand or your_seat not in {0, 1} or button not in {0, 1}:
        return False

    future_cost = 0
    future_button = 1 - button
    for _ in range(hand + 1, total + 1):
        future_cost += small_blind if future_button == your_seat else big_blind
        future_button = 1 - future_button
    return live_stack - future_cost >= starting_stack + targets[phase]


def _clear_rule_memory_for_tests() -> None:
    with _RULE_MEMORY_LOCK:
        _RULE_MEMORY.clear()


def _first_legal(legal: set[Action], *preferences: Action) -> Move:
    for action in preferences:
        if action in legal:
            return {"action": action}
    for action in ("check", "call", "fold"):
        if action in legal:
            return {"action": action}
    # Valid game states never reach this branch, but a legal bet/raise-only probe
    # is still handled by the caller's range-aware aggressive path.
    return {"action": next(iter(legal), "check")}


def _can_aggress(legal: set[Action]) -> bool:
    return bool(legal & AGGRESSIVE_ACTIONS)


def _pot_odds(to_call: int, pot: int) -> float:
    return to_call / max(1, pot + to_call)


def _mix(state: Mapping[str, Any], purpose: str) -> float:
    source = "|".join(
        (
            str(state.get("match_id", "")),
            str(state.get("hand_number", "")),
            str(state.get("round", "")),
            str(len(_actions(state.get("current_hand_actions")))),
            purpose,
        )
    )
    digest = hashlib.blake2s(source.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / (2**64 - 1)


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _bounded_int(value: Any, low: int, high: int, *, default: int) -> int:
    parsed = _int(value, default)
    return max(low, min(high, parsed))


def _optional_bounded_int(value: Any, low: int, high: int) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(low, min(high, value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
