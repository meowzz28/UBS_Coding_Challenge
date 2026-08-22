"""General SHOWDOWN Phase 1-3 betting strategy and HTTP routes."""

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
PRIME_NUMBERS = frozenset({2, 3, 5, 7, 11, 13})
FEATURE_MODEL_NAME = "feature-linear-v1"


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
    opponents: int = 1


@dataclass(frozen=True)
class LearnedRuleModel:
    """A deterministic pairwise preference model for one opaque table rule."""

    weights: tuple[float, ...]
    confidence: float
    observations: int
    training_fit: float


_RULE_MEMORY: dict[
    str, dict[tuple[str, int, int, int, int, int, int, int], ShowdownObservation]
] = {}
_RULE_MODEL_CACHE: dict[
    str, tuple[tuple[ShowdownObservation, ...], LearnedRuleModel]
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

    # At the start of an opaque-rule leg, reach inexpensive showdowns to learn
    # instead of risking chips based on an assumed ordering. A multiway showdown
    # contributes several comparisons, so Phase 3 needs only a few such hands.
    if (
        _int(state.get("phase"), 0) in {2, 3}
        and equity.observations < (12 if equity.opponents == 1 else 18)
        and (
            equity.observations < (9 if equity.opponents == 1 else 10)
            or equity.confidence < 0.76
        )
    ):
        return _learning_move(
            state,
            legal,
            to_call,
            pot,
            stack,
            equity.equity,
        )

    if _int(state.get("phase"), 0) == 3:
        return _phase_three_move(
            state=state,
            legal=legal,
            to_call=to_call,
            pot=pot,
            stack=stack,
            own_bet=own_bet,
            profile=profile,
            equity=equity.equity,
            opponents=equity.opponents,
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
        return EquityEstimate(equity, 1.0, 100, "standard", 1)

    rule_name = str(state.get("table_rule", ""))
    _remember_showdowns(state, rule_name)
    observations = _rule_observations(rule_name)
    learned = _learn_rule_model(rule_name, observations)
    opponents = _live_opponent_count(state)

    if community is None:
        total = (
            sum(
                _feature_multiway_equity(learned, your_number, revealed, opponents)
                for revealed in range(1, 14)
            )
            / 13
        )
    else:
        total = _feature_multiway_equity(learned, your_number, community, opponents)
    return EquityEstimate(
        total,
        learned.confidence,
        learned.observations,
        FEATURE_MODEL_NAME,
        opponents,
    )


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
    additions: dict[
        tuple[str, int, int, int, int, int, int, int], ShowdownObservation
    ] = {}
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
        if len(seats) < 2:
            continue
        seats.sort()
        winner_seats: set[int] = set()
        for winner in winners:
            if isinstance(winner, int) and not isinstance(winner, bool):
                winner_seats.add(winner)
        if not winner_seats:
            continue

        # Winners are known to tie one another and to outrank every shown
        # non-winner. The relative order of two losing hands is not observable,
        # so deliberately do not manufacture a label for such a pair.
        for first_index, (first_seat, first_number) in enumerate(seats):
            for second_seat, second_number in seats[first_index + 1 :]:
                first_won = first_seat in winner_seats
                second_won = second_seat in winner_seats
                if first_won and second_won:
                    result = 0
                elif first_won:
                    result = 1
                elif second_won:
                    result = -1
                else:
                    continue
                key = (
                    match_id,
                    leg_number,
                    hand_number,
                    first_seat,
                    second_seat,
                    first_number,
                    second_number,
                    community,
                )
                additions[key] = ShowdownObservation(
                    first_number, second_number, community, result
                )

    if additions:
        with _RULE_MEMORY_LOCK:
            memory = _RULE_MEMORY.setdefault(rule_name, {})
            changed = any(memory.get(key) != value for key, value in additions.items())
            memory.update(additions)
            while len(memory) > 500:
                memory.pop(next(iter(memory)))
                changed = True
            if changed:
                _RULE_MODEL_CACHE.pop(rule_name, None)


def _rule_observations(rule_name: str) -> tuple[ShowdownObservation, ...]:
    with _RULE_MEMORY_LOCK:
        observations = tuple(_RULE_MEMORY.get(rule_name, {}).values())
    return tuple(
        sorted(
            observations,
            key=lambda item: (
                item.community,
                item.first_number,
                item.second_number,
                item.result,
            ),
        )
    )


def _learn_rule_model(
    rule_name: str,
    observations: Sequence[ShowdownObservation],
) -> LearnedRuleModel:
    signature = tuple(observations)
    with _RULE_MEMORY_LOCK:
        cached = _RULE_MODEL_CACHE.get(rule_name)
        if cached is not None and cached[0] == signature:
            return cached[1]

    feature_count = len(_rule_features(1, 1))
    rows: list[tuple[tuple[float, ...], int]] = []
    for observation in signature:
        first = _rule_features(observation.first_number, observation.community)
        second = _rule_features(observation.second_number, observation.community)
        difference = tuple(left - right for left, right in zip(first, second))
        if any(abs(value) > 1e-12 for value in difference):
            rows.append((difference, observation.result))

    if not rows:
        model = LearnedRuleModel((0.0,) * feature_count, 0.0, 0, 0.5)
    else:
        # Batch logistic ranking is deterministic because the complete gradient is
        # applied at once. A showdown says whether score(first, community) should
        # be above, below, or level with score(second, community); no named rule
        # or fixed catalogue is involved.
        weights = [0.0] * feature_count
        for epoch in range(100):
            learning_rate = 0.45 / (1.0 + 0.035 * epoch)
            gradient = [0.0] * feature_count
            for difference, result in rows:
                margin = sum(
                    weight * value
                    for weight, value in zip(weights, difference, strict=True)
                )
                probability = _sigmoid(margin)
                target = 0.5 if result == 0 else 0.99 if result > 0 else 0.01
                error = target - probability
                for index, value in enumerate(difference):
                    gradient[index] += error * value

            scale = learning_rate / len(rows)
            decay = 1.0 - 0.02 * learning_rate
            for index in range(feature_count):
                updated = decay * weights[index] + scale * gradient[index]
                # Proximal L1 regularisation suppresses accidental correlations
                # from a tiny sample. Context/local features pay a slightly larger
                # complexity cost than the broadly reusable mathematical features.
                l1_cost = 0.06 * (1.0 if index < 35 else 1.5)
                threshold = l1_cost * learning_rate
                if updated > threshold:
                    weights[index] = updated - threshold
                elif updated < -threshold:
                    weights[index] = updated + threshold
                else:
                    weights[index] = 0.0

        fit = _training_fit(weights, rows)
        distinct_numbers = {
            number
            for observation in signature
            for number in (observation.first_number, observation.second_number)
        }
        distinct_communities = {observation.community for observation in signature}
        coverage = min(1.0, len(rows) / 12.0)
        diversity = min(
            1.0,
            0.5 * len(distinct_numbers) / 9.0 + 0.5 * len(distinct_communities) / 7.0,
        )
        learned_signal = _clamp((fit - 0.5) / 0.42, 0.0, 1.0)
        confidence = coverage * (0.76 * learned_signal + 0.24 * diversity)
        model = LearnedRuleModel(
            tuple(weights),
            _clamp(confidence, 0.0, 1.0),
            len(rows),
            fit,
        )

    with _RULE_MEMORY_LOCK:
        _RULE_MODEL_CACHE[rule_name] = (signature, model)
    return model


def _rule_features(number: int, community: int) -> tuple[float, ...]:
    """Describe a card without assuming any particular secret table rule."""

    number_scaled = (number - 7) / 6
    community_scaled = (community - 7) / 6
    distance = abs(number - community)
    distance_scaled = distance / 12
    signed_distance = (number - community) / 12
    clockwise = (number - community) % 13
    odd = 1.0 if number % 2 else -1.0
    prime = 1.0 if number in PRIME_NUMBERS else -1.0
    community_half = 1.0 if community <= 7 else -1.0
    community_parity = 1.0 if community % 2 else -1.0
    community_prime = 1.0 if community in PRIME_NUMBERS else -1.0

    features = [
        5.0 * number_scaled,
        2.0 * number_scaled**2,
        1.5 * number_scaled**3,
        3.0 * odd,
        3.0 * prime,
        10.0 if number == community else 0.0,
        5.0 * distance_scaled,
        2.0 * distance_scaled**2,
        2.0 * signed_distance,
        1.2 if number > community else -1.2 if number < community else 0.0,
        2.0 if number % 2 == community % 2 else -2.0,
        2.0 if (number in PRIME_NUMBERS) == (community in PRIME_NUMBERS) else -2.0,
        4.0 * clockwise / 12,
        4.0 * ((community - number) % 13) / 12,
        4.0 * abs(number - 7) / 6,
    ]

    # Interactions let the learned ordering change with the community number.
    # These gates are generic mathematical properties, not hypotheses about
    # which hidden rules the coordinator may use.
    base = (number_scaled, odd, prime, distance_scaled, signed_distance)
    for gate in (
        community_scaled,
        community_half,
        community_parity,
        community_prime,
    ):
        features.extend(2.20 * value * gate for value in base)

    # Cumulative bases interpolate rankings between sparse early showdowns;
    # compact local bases retain enough flexibility for non-monotonic rules.
    features.extend(0.34 * (number >= level) for level in range(2, 14))
    features.extend(0.34 * (distance >= level) for level in range(1, 13))
    features.extend(0.24 * (clockwise >= level) for level in range(1, 13))
    features.extend(0.20 * (number - community >= level) for level in range(1, 13))
    features.extend(0.20 * (community - number >= level) for level in range(1, 13))
    features.extend(0.16 * (number == level) for level in range(1, 14))
    features.extend(0.14 * (distance == level) for level in range(13))
    for gate in (community_half, community_parity, community_prime):
        features.extend(0.16 * (number >= level) * gate for level in range(2, 14))
    return tuple(float(value) for value in features)


def _feature_score(model: LearnedRuleModel, number: int, community: int) -> float:
    return sum(
        weight * value
        for weight, value in zip(
            model.weights,
            _rule_features(number, community),
            strict=True,
        )
    )


def _feature_equity(
    model: LearnedRuleModel,
    your_number: int,
    community: int,
) -> float:
    return _feature_multiway_equity(model, your_number, community, 1)


def _feature_multiway_equity(
    model: LearnedRuleModel,
    your_number: int,
    community: int,
    opponents: int,
) -> float:
    """Expected pot share against independent uniform live opponents."""

    opponents = max(1, opponents)
    fair_share = 1.0 / (opponents + 1)
    if model.observations == 0:
        return fair_share

    your_score = _feature_score(model, your_number, community)
    opponent_scores = [
        _feature_score(model, opponent, community) for opponent in range(1, 14)
    ]
    spread = max(opponent_scores) - min(opponent_scores)
    tie_tolerance = max(1e-8, spread * 0.008)
    wins = 0
    ties = 0
    for opponent_score in opponent_scores:
        difference = your_score - opponent_score
        if difference > tie_tolerance:
            wins += 1
        elif difference >= -tie_tolerance:
            ties += 1

    win_probability = wins / 13
    tie_probability = ties / 13
    # We receive 1/(k+1) of the pot when exactly k opponents tie us and every
    # other opponent ranks below us. Any opponent above us contributes zero.
    raw_equity = sum(
        math.comb(opponents, tied)
        * tie_probability**tied
        * win_probability ** (opponents - tied)
        / (tied + 1)
        for tied in range(opponents + 1)
    )
    reliability = 0.25 + 0.75 * model.confidence
    return _clamp(
        fair_share + (raw_equity - fair_share) * reliability,
        0.0,
        1.0,
    )


def _training_fit(
    weights: Sequence[float],
    rows: Sequence[tuple[tuple[float, ...], int]],
) -> float:
    quality = 0.0
    for difference, result in rows:
        margin = sum(
            weight * value for weight, value in zip(weights, difference, strict=True)
        )
        probability = _sigmoid(margin)
        if result > 0:
            quality += probability
        elif result < 0:
            quality += 1.0 - probability
        else:
            quality += 1.0 - 2.0 * abs(probability - 0.5)
    return quality / len(rows)


def _sigmoid(value: float) -> float:
    bounded = _clamp(value, -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-bounded))


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
    phase = _int(state.get("phase"), 0)
    if phase == 3 and round_name == "pre_reveal":
        learning_cap = max(
            3,
            min(round(pot * 0.65), round(stack * 0.04), 8),
        )
    elif phase == 3:
        learning_cap = max(
            4,
            min(round(pot * 0.60), round(stack * 0.06), 10),
        )
    elif round_name == "pre_reveal":
        learning_cap = max(
            5,
            min(round(pot * 1.00), round(stack * 0.09), 14),
        )
    else:
        learning_cap = max(
            7,
            min(round(pot * 1.05), round(stack * 0.11), 18),
        )
    risk_limit = 0.08 if phase == 3 else 0.13
    informative_call = to_call <= learning_cap and risk <= risk_limit
    priced_call = risk <= (0.10 if phase == 3 else 0.15) and equity >= price - 0.04
    if "call" in legal and (blind_completion or informative_call or priced_call):
        return {"action": "call"}
    return _first_legal(legal, "fold", "call", "check")


def _phase_three_move(
    *,
    state: Mapping[str, Any],
    legal: set[Action],
    to_call: int,
    pot: int,
    stack: int,
    own_bet: int,
    profile: OpponentProfile,
    equity: float,
    opponents: int,
) -> Move:
    """Choose a six-seat action using actual expected multiway pot share."""

    fair_share = 1.0 / (opponents + 1)
    score_bias = _phase_three_score_bias(state)
    adjusted = _clamp(equity + score_bias, 0.0, 1.0)
    protecting = score_bias < -0.025
    round_name = str(state.get("round", "pre_reveal"))

    # A qualifying late lead is worth more than a thin positive-chip EV wager.
    # Still play genuinely premium holdings because opponents can overtake one
    # another even after we fold.
    protect_threshold = max(0.42, fair_share * 2.7)
    if protecting and adjusted < protect_threshold:
        if to_call > 0:
            return _first_legal(legal, "fold", "call", "check")
        return _first_legal(legal, "check", "call", "fold")

    if round_name == "post_reveal":
        premium = max(0.34, fair_share * 2.15)
        near_nuts = max(0.55, fair_share * 3.35)
        if to_call > 0:
            price = _pot_odds(to_call, pot)
            risk = to_call / max(1, stack)
            urgent = score_bias > 0.025
            risk_limit = 0.34 if urgent else 0.20
            if adjusted >= near_nuts and risk <= 0.30 and _can_aggress(legal):
                return _aggressive_move(state, legal, own_bet, to_call, pot, 0.62)
            tendency_adjustment = (profile.aggression_rate - 0.30) * 0.10
            call_equity = adjusted + tendency_adjustment
            margin = 0.015 if urgent else 0.04
            if "call" in legal and risk <= risk_limit and call_equity >= price + margin:
                return {"action": "call"}
            return _first_legal(legal, "fold", "call", "check")

        if adjusted >= near_nuts and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, 0, pot, 0.72)
        if adjusted >= premium and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, 0, pot, 0.46)
        if adjusted <= fair_share * 0.55 and _can_aggress(legal):
            bluff_rate = _clamp(
                0.025 + (profile.fold_to_aggression - 0.34) * 0.25,
                0.01,
                0.11,
            )
            if _mix(state, "multiway-bluff") < bluff_rate:
                return _aggressive_move(state, legal, own_bet, 0, pot, 0.34)
        return _first_legal(legal, "check", "call", "fold")

    actions = _actions(state.get("current_hand_actions"))
    has_raise = any(
        str(action.get("action", "")) in AGGRESSIVE_ACTIONS for action in actions
    )
    cheap_entry = to_call <= max(2, _int(state.get("big_blind"), 2)) and not has_raise
    premium = max(0.30, fair_share * 1.90)
    playable = max(0.13, fair_share * 0.82)
    if to_call > 0:
        price = _pot_odds(to_call, pot)
        risk = to_call / max(1, stack)
        if adjusted >= premium and risk <= 0.14 and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 0.44)
        if cheap_entry and adjusted >= playable and "call" in legal:
            return {"action": "call"}
        risk_limit = 0.20 if score_bias > 0.025 else 0.12
        if "call" in legal and risk <= risk_limit and adjusted >= price + 0.035:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")

    if adjusted >= premium and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.42)
    return _first_legal(legal, "check", "call", "fold")


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
    current_actions = _actions(state.get("current_hand_actions"))
    target_seat: int | None = None
    for action in reversed(current_actions):
        seat = _int(action.get("seat"), -2)
        if seat != your_seat and str(action.get("action", "")) in AGGRESSIVE_ACTIONS:
            target_seat = seat
            break

    def tracked(seat: int) -> bool:
        return seat != your_seat and (target_seat is None or seat == target_seat)

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
                if tracked(seat):
                    opponent_actions += 1
                    if action_name in AGGRESSIVE_ACTIONS:
                        opponent_aggression += 1

                if (
                    seat == your_seat
                    and action_name in AGGRESSIVE_ACTIONS
                    and index + 1 < len(round_actions)
                ):
                    answered_seats: set[int] = set()
                    for answer in round_actions[index + 1 :]:
                        answer_seat = _int(answer.get("seat"), -2)
                        if answer_seat == your_seat:
                            break
                        if answer_seat in answered_seats or not tracked(answer_seat):
                            continue
                        answered_seats.add(answer_seat)
                        responses += 1
                        if str(answer.get("action", "")) == "fold":
                            folds += 1

    # Include this hand for aggression estimation but not fold response—the hand
    # is incomplete and a missing next action is not evidence of a fold.
    for action in current_actions:
        if tracked(_int(action.get("seat"), -2)):
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

    phase = _int(state.get("phase"), 1)
    if phase == 2:
        pot_fraction *= 1.20
    elif phase == 3:
        pot_fraction *= 0.90
    desired = own_bet + to_call + max(1, round(pot * pot_fraction))
    if phase in {2, 3}:
        # Later phases are scored on thresholds, not on maximizing a lucky stack.
        # Never let a large minimum raise silently turn an ordinary value line
        # into a quarter-match-or-more wager.
        live_stack = max(0, _int(state.get("your_stack"), 0))
        extra_cap = max(4, round(live_stack * (0.24 if phase == 2 else 0.18)))
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


def _live_opponent_count(state: Mapping[str, Any]) -> int:
    """Count opponents still eligible to win the current hand."""

    your_seat = _int(state.get("your_seat"), -1)
    players = state.get("players")
    count = 0
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        for player in players:
            if not isinstance(player, Mapping):
                continue
            if _int(player.get("seat"), -2) == your_seat:
                continue
            if player.get("folded") is True or player.get("busted") is True:
                continue
            count += 1
    # A valid turn always has at least one live opponent. The fallback keeps
    # partial protocol probes conservative and preserves Phase 2 behaviour.
    return max(1, count)


def _phase_three_score_bias(state: Mapping[str, Any]) -> float:
    """Trade chip EV for the +10-and-strictly-first objective late in a leg."""

    players = state.get("players")
    your_seat = _int(state.get("your_seat"), -1)
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return 0.0

    own_delta: int | None = None
    opponent_deltas: list[int] = []
    for player in players:
        if not isinstance(player, Mapping):
            continue
        delta = _int(player.get("chip_delta"), 0)
        if _int(player.get("seat"), -2) == your_seat:
            own_delta = delta
        else:
            opponent_deltas.append(delta)
    if own_delta is None or not opponent_deltas:
        return 0.0

    hand = _int(state.get("hand_number"), 0)
    total = _int(state.get("total_hands"), 0)
    if hand < 1 or total < 1:
        return 0.0
    progress = _clamp(hand / total, 0.0, 1.0)
    if progress < 0.55:
        return 0.0

    lead = own_delta - max(opponent_deltas)
    urgency = max(0.0, 10 - own_delta) + max(0.0, 1 - lead)
    if urgency > 0:
        return _clamp(0.012 + urgency / 500 * progress, 0.0, 0.06)
    if progress >= 0.72 and own_delta >= 10 and lead >= 8:
        return -_clamp(0.02 + lead / 600 * progress, 0.02, 0.055)
    return 0.0


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
        _RULE_MODEL_CACHE.clear()


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
