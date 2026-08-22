"""General SHOWDOWN Phase 1-3 betting strategy and HTTP routes."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, TypedDict, cast

from fastapi import APIRouter

Action = Literal["check", "call", "bet", "raise", "fold"]
AGGRESSIVE_ACTIONS = {"bet", "raise"}
PRIME_NUMBERS = frozenset({2, 3, 5, 7, 11, 13})
EXACT_MODEL_NAME = "exact-hypothesis-v2"
# Compatibility export retained for callers that displayed the old model name.
FEATURE_MODEL_NAME = EXACT_MODEL_NAME
RankKey = tuple[float, ...]
Ranker = Callable[[int, int], RankKey]


def _target_under_rank(number: int, community: int, target: int) -> RankKey:
    total = number + community
    return (1.0, float(total)) if total <= target else (0.0, float(-total))


def _target_nearest_rank(number: int, community: int, target: int) -> RankKey:
    total = number + community
    return (float(-abs(total - target)),)


RULE_HYPOTHESES: dict[str, Ranker] = {
    "standard": lambda number, community: (float(number == community), float(number)),
    "high": lambda number, _community: (float(number),),
    "pair-low": lambda number, community: (
        float(number == community),
        float(-number),
    ),
    "low": lambda number, _community: (float(-number),),
    "proximity": lambda number, community: (float(-abs(number - community)),),
    "proximity-high": lambda number, community: (
        float(-abs(number - community)),
        float(number),
    ),
    "proximity-low": lambda number, community: (
        float(-abs(number - community)),
        float(-number),
    ),
    "anti-proximity": lambda number, community: (float(abs(number - community)),),
    "anti-proximity-high": lambda number, community: (
        float(abs(number - community)),
        float(number),
    ),
    "anti-proximity-low": lambda number, community: (
        float(abs(number - community)),
        float(-number),
    ),
    "clockwise": lambda number, community: (float(-((number - community) % 13)),),
    "counter-clockwise": lambda number, community: (
        float(-((community - number) % 13)),
    ),
    "clockwise-farthest": lambda number, community: (float((number - community) % 13),),
    "counter-clockwise-farthest": lambda number, community: (
        float((community - number) % 13),
    ),
    "near-seven": lambda number, _community: (
        float(-abs(number - 7)),
        float(number),
    ),
    "community-reversal": lambda number, community: (
        float(number if community <= 7 else -number),
    ),
    "odd-high": lambda number, _community: (float(number % 2), float(number)),
    "prime-high": lambda number, _community: (
        float(number in PRIME_NUMBERS),
        float(number),
    ),
    "target-14-under": lambda number, community: _target_under_rank(
        number, community, 14
    ),
    "target-21-under": lambda number, community: _target_under_rank(
        number, community, 21
    ),
    "target-14-nearest": lambda number, community: _target_nearest_rank(
        number, community, 14
    ),
    "target-14-nearest-high": lambda number, community: (
        *_target_nearest_rank(number, community, 14),
        float(number + community),
    ),
    "target-14-nearest-low": lambda number, community: (
        *_target_nearest_rank(number, community, 14),
        float(-(number + community)),
    ),
    "target-21-nearest": lambda number, community: _target_nearest_rank(
        number, community, 21
    ),
    "target-21-nearest-high": lambda number, community: (
        *_target_nearest_rank(number, community, 21),
        float(number + community),
    ),
    "target-21-nearest-low": lambda number, community: (
        *_target_nearest_rank(number, community, 21),
        float(-(number + community)),
    ),
}


class Move(TypedDict, total=False):
    action: Action
    amount: int


@dataclass(frozen=True)
class OpponentProfile:
    """Persistent per-player tendencies with conservative Bayesian smoothing."""

    fold_to_aggression: float
    aggression_rate: float
    observations: int
    name: str = "unknown"
    vpip: float = 0.5
    pfr: float = 0.25
    aggression_factor: float = 1.0
    fold_to_raise: float = 0.33
    busted: bool = False
    archetype: str = "unknown"


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
    percentile: float = 0.5
    locked_rule: str | None = None
    robust_equity: float = 0.0
    robust_percentile: float = 0.0
    premium_support: float = 0.0


@dataclass(frozen=True)
class ExactRuleModel:
    """An exact hypothesis lock or a transitive pairwise fallback model."""

    candidates: tuple[str, ...]
    locked_rule: str | None
    relations: tuple[tuple[tuple[int | None, ...], ...], ...]
    confidence: float
    observations: int


@dataclass(frozen=True)
class OpponentHandObservation:
    vpip: bool
    pfr: bool
    aggressive_actions: int
    calls: int
    faced_raise: int
    folded_to_raise: int


_RULE_MEMORY: dict[
    str, dict[tuple[str, int, int, int, int, int, int, int], ShowdownObservation]
] = {}
_RULE_MODEL_CACHE: dict[
    str, tuple[tuple[ShowdownObservation, ...], ExactRuleModel]
] = {}
_OPPONENT_MEMORY: dict[str, dict[tuple[str, int, int], OpponentHandObservation]] = {}
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

    if _int(state.get("phase"), 0) == 3:
        return _phase_three_move(
            state=state,
            legal=legal,
            to_call=to_call,
            pot=pot,
            stack=stack,
            own_bet=own_bet,
            profile=profile,
            estimate=equity,
        )

    # Phase 2 buys a bounded number of cheap showdowns until the exact engine
    # locks a rule or the pairwise fallback has enough coverage.
    if (
        _int(state.get("phase"), 0) == 2
        and equity.locked_rule is None
        and equity.observations < 12
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
        return EquityEstimate(
            equity,
            1.0,
            100,
            "standard",
            1,
            equity,
            "standard",
            equity,
            equity,
            float(equity >= 0.75),
        )

    rule_name = str(state.get("table_rule", ""))
    _remember_showdowns(state, rule_name)
    observations = _rule_observations(rule_name)
    learned = _learn_rule_model(rule_name, observations)
    opponents = _live_opponent_count(state)

    if phase != 3 and learned.observations == 0:
        fair_share = 1.0 / (opponents + 1)
        return EquityEstimate(
            fair_share,
            learned.confidence,
            0,
            EXACT_MODEL_NAME,
            opponents,
            0.5,
            None,
            fair_share,
            0.5,
            0.0,
        )

    if learned.candidates:
        # Preserve each rule as one coherent world while averaging across the
        # unknown community number. Taking a minimum independently on every
        # community mixes mutually exclusive rules and is needlessly pessimistic.
        by_rule: list[tuple[float, float]] = []
        for candidate in learned.candidates:
            ranker = RULE_HYPOTHESES[candidate]
            if community is None:
                values = [
                    _ranker_equity(ranker, your_number, revealed, opponents)
                    for revealed in range(1, 14)
                ]
                by_rule.append(
                    (
                        sum(value[0] for value in values) / 13,
                        sum(value[1] for value in values) / 13,
                    )
                )
            else:
                by_rule.append(
                    _ranker_equity(ranker, your_number, community, opponents)
                )
        total = sum(item[0] for item in by_rule) / len(by_rule)
        percentile = sum(item[1] for item in by_rule) / len(by_rule)
        robust_equity = _lower_quartile(item[0] for item in by_rule)
        robust_percentile = _lower_quartile(item[1] for item in by_rule)
        premium_support = sum(item[1] >= 0.75 for item in by_rule) / len(by_rule)
    else:
        if community is None:
            estimates = [
                _exact_multiway_equity(learned, your_number, revealed, opponents)
                for revealed in range(1, 14)
            ]
            total = sum(item[0] for item in estimates) / 13
            percentile = sum(item[1] for item in estimates) / 13
        else:
            total, percentile = _exact_multiway_equity(
                learned, your_number, community, opponents
            )
        robust_equity = total
        robust_percentile = percentile
        premium_support = float(percentile >= 0.75)
    return EquityEstimate(
        total,
        learned.confidence,
        learned.observations,
        EXACT_MODEL_NAME,
        opponents,
        percentile,
        learned.locked_rule,
        robust_equity,
        robust_percentile,
        premium_support,
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
) -> ExactRuleModel:
    signature = tuple(observations)
    with _RULE_MEMORY_LOCK:
        cached = _RULE_MODEL_CACHE.get(rule_name)
        if cached is not None and cached[0] == signature:
            return cached[1]

    candidates = tuple(
        name
        for name, ranker in RULE_HYPOTHESES.items()
        if all(_rank_result(ranker, item) == item.result for item in signature)
    )
    relations = _build_transitive_relations(signature)
    locked = candidates[0] if len(candidates) == 1 else None
    if not signature:
        confidence = 0.0
    elif locked is not None:
        confidence = 1.0
    elif candidates:
        eliminated = 1.0 - len(candidates) / len(RULE_HYPOTHESES)
        confidence = _clamp(0.25 + 0.55 * eliminated + len(signature) / 100, 0.0, 0.94)
    else:
        confidence = _clamp(0.30 + len(signature) / 50, 0.0, 0.88)
    model = ExactRuleModel(
        candidates,
        locked,
        relations,
        confidence,
        len(signature),
    )

    with _RULE_MEMORY_LOCK:
        _RULE_MODEL_CACHE[rule_name] = (signature, model)
    return model


def _rank_result(ranker: Ranker, observation: ShowdownObservation) -> int:
    first = ranker(observation.first_number, observation.community)
    second = ranker(observation.second_number, observation.community)
    return (first > second) - (first < second)


def _build_transitive_relations(
    observations: Sequence[ShowdownObservation],
) -> tuple[tuple[tuple[int | None, ...], ...], ...]:
    """Build direct-majority relations and safe strict transitive closure."""

    counts: dict[tuple[int, int, int], list[int]] = {}
    for item in observations:
        key = (item.community - 1, item.first_number - 1, item.second_number - 1)
        bucket = counts.setdefault(key, [0, 0, 0])
        bucket[0 if item.result > 0 else 1 if item.result < 0 else 2] += 1
        reverse = (key[0], key[2], key[1])
        reverse_bucket = counts.setdefault(reverse, [0, 0, 0])
        reverse_bucket[1 if item.result > 0 else 0 if item.result < 0 else 2] += 1

    cube: list[list[list[int | None]]] = []
    for community in range(13):
        matrix: list[list[int | None]] = [[None] * 13 for _ in range(13)]
        for card in range(13):
            matrix[card][card] = 0
        for (seen_community, first, second), bucket in counts.items():
            if seen_community != community:
                continue
            best = max(bucket)
            winners = [index for index, value in enumerate(bucket) if value == best]
            if len(winners) == 1:
                matrix[first][second] = (1, -1, 0)[winners[0]]

        reach = [
            [matrix[first][second] == 1 for second in range(13)] for first in range(13)
        ]
        for middle in range(13):
            for first in range(13):
                if not reach[first][middle]:
                    continue
                for second in range(13):
                    reach[first][second] = reach[first][second] or reach[middle][second]
        for first in range(13):
            for second in range(13):
                if matrix[first][second] is not None:
                    continue
                if reach[first][second] and not reach[second][first]:
                    matrix[first][second] = 1
                elif reach[second][first] and not reach[first][second]:
                    matrix[first][second] = -1
        cube.append(matrix)
    return tuple(
        tuple(tuple(row) for row in community_matrix) for community_matrix in cube
    )


def _share_from_counts(wins: float, ties: float, opponents: int) -> float:
    win_probability = wins / 13
    tie_probability = ties / 13
    return sum(
        math.comb(opponents, tied)
        * tie_probability**tied
        * win_probability ** (opponents - tied)
        / (tied + 1)
        for tied in range(opponents + 1)
    )


def _ranker_equity(
    ranker: Ranker,
    your_number: int,
    community: int,
    opponents: int,
) -> tuple[float, float]:
    your_rank = ranker(your_number, community)
    ranks = [ranker(number, community) for number in range(1, 14)]
    wins = sum(rank < your_rank for rank in ranks)
    ties = sum(rank == your_rank for rank in ranks)
    return _share_from_counts(wins, ties, opponents), (wins + 0.5 * ties) / 13


def _exact_multiway_equity(
    model: ExactRuleModel,
    your_number: int,
    community: int,
    opponents: int,
) -> tuple[float, float]:
    """Return expected pot share and exact/empirical single-opponent percentile."""

    opponents = max(1, opponents)
    fair_share = 1.0 / (opponents + 1)
    if model.candidates:
        values = [
            _ranker_equity(RULE_HYPOTHESES[name], your_number, community, opponents)
            for name in model.candidates
        ]
        return (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )

    row = model.relations[community - 1][your_number - 1]
    wins = sum(result == 1 for result in row)
    losses = sum(result == -1 for result in row)
    ties = sum(result == 0 for result in row)
    unknown = 13 - wins - losses - ties
    known = wins + losses + ties
    if known <= 1:
        return fair_share, 0.5
    known_percentile = (wins + 0.5 * ties) / known
    estimated_wins = wins + unknown * known_percentile
    raw = _share_from_counts(estimated_wins, ties, opponents)
    reliability = 0.35 + 0.65 * model.confidence
    return (
        fair_share + (raw - fair_share) * reliability,
        0.5 + (known_percentile - 0.5) * reliability,
    )


def _lower_quartile(values: Iterable[float]) -> float:
    """Return a deterministic lower quartile without interpolation."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[(len(ordered) - 1) // 4]


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
    estimate: EquityEstimate,
) -> Move:
    """High-variance EV maximizer strictly tuned for Phase 3 leaderboard dominance."""

    equity = estimate.equity
    percentile = estimate.percentile
    round_name = str(state.get("round", "pre_reveal"))
    hand = max(1, _int(state.get("hand_number"), 1))
    total_hands = max(hand, _int(state.get("total_hands"), 60))
    
    own_delta, leader_delta, leader_seat = _leaderboard(state)
    deficit = max(0, leader_delta - own_delta + 1)
    
    target_seat = _last_aggressor_seat(state)
    leader_exposed = _leader_is_exposed(state, leader_seat, target_seat)
    
    # 1. THE GUARANTEED LOCK
    if _strict_majority_locked(state, stack, own_delta) or _final_hand_rank_locked(state, stack, own_delta):
        if to_call > 0:
            return _first_legal(legal, "fold", "call", "check")
        return _first_legal(legal, "check", "call", "fold")

    # 2. KAMIKAZE EXPLORATION (Hands 1-15)
    if estimate.locked_rule is None and hand <= 15:
        if to_call <= max(10, stack // 10) and "call" in legal:
            return {"action": "call"}
        if to_call == 0 and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, 0, pot, 0.5)

    # 3. ENDGAME DESPERATION (Hands 45-60)
    if hand >= 45 and deficit > 0:
        if percentile >= 0.75 and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 2.0, all_in=True)
        if percentile >= 0.65 and to_call > 0 and "call" in legal:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")

    # 4. EXPLOITATIVE LEADER TARGETING
    if leader_exposed and to_call > 0:
        if percentile >= 0.85 and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 1.25)
        if percentile >= 0.68 and "call" in legal:
            return {"action": "call"}

    # 5. CORE EXPLOITATIVE VALUE (Hands 16-44)
    price = _pot_odds(to_call, pot)
    
    if round_name == "post_reveal":
        if to_call > 0:
            if percentile >= 0.90 and _can_aggress(legal):
                return _aggressive_move(state, legal, own_bet, to_call, pot, 1.0)
            if equity >= price + 0.02 and "call" in legal:
                return {"action": "call"}
            return _first_legal(legal, "fold", "call", "check")
        else:
            if percentile >= 0.80 and _can_aggress(legal):
                return _aggressive_move(state, legal, own_bet, 0, pot, 0.75)
            if percentile < 0.30 and profile.fold_to_raise > 0.60 and _can_aggress(legal):
                return _aggressive_move(state, legal, own_bet, 0, pot, 0.50)
            return _first_legal(legal, "check", "call", "fold")

    if to_call > 0:
        if percentile >= 0.92 and _can_aggress(legal):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 1.0)
        if equity >= price + 0.05 and "call" in legal:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")
    
    if percentile >= 0.82 and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.60)
        
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
    _remember_opponents(state)
    players = _player_maps(state)
    your_seat = _int(state.get("your_seat"), -1)
    target = _last_aggressor_seat(state)
    if target is not None and target in players:
        name, busted = players[target]
        return _profile_for_name(name, busted)

    profiles = [
        _profile_for_name(name, busted)
        for seat, (name, busted) in players.items()
        if seat != your_seat and not busted
    ]
    if not profiles:
        return OpponentProfile(1 / 3, 0.30, 0)
    return OpponentProfile(
        sum(item.fold_to_aggression for item in profiles) / len(profiles),
        sum(item.aggression_rate for item in profiles) / len(profiles),
        sum(item.observations for item in profiles),
        "table",
        sum(item.vpip for item in profiles) / len(profiles),
        sum(item.pfr for item in profiles) / len(profiles),
        sum(item.aggression_factor for item in profiles) / len(profiles),
        sum(item.fold_to_raise for item in profiles) / len(profiles),
        False,
        "mixed",
    )


def _player_maps(state: Mapping[str, Any]) -> dict[int, tuple[str, bool]]:
    result: dict[int, tuple[str, bool]] = {}
    players = state.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return result
    for player in players:
        if not isinstance(player, Mapping):
            continue
        seat = _int(player.get("seat"), -1)
        if seat >= 0:
            result[seat] = (
                str(player.get("name", f"seat-{seat}")),
                player.get("busted") is True,
            )
    return result


def _remember_opponents(state: Mapping[str, Any]) -> None:
    players = _player_maps(state)
    your_seat = _int(state.get("your_seat"), -1)
    histories = state.get("recent_hands")
    if not isinstance(histories, Sequence) or isinstance(histories, (str, bytes)):
        return
    match_id = str(state.get("match_id", ""))
    leg = _int(state.get("leg_number"), 0)
    additions: list[tuple[str, tuple[str, int, int], OpponentHandObservation]] = []
    for hand in histories:
        if not isinstance(hand, Mapping):
            continue
        hand_number = _int(hand.get("hand_number"), 0)
        if hand_number < 1:
            continue
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for action in _actions(hand.get("actions")):
            grouped.setdefault(str(action.get("round", "")), []).append(action)
        for seat, (name, _busted) in players.items():
            if seat == your_seat:
                continue
            pre = grouped.get("pre_reveal", [])
            own_pre = [item for item in pre if _int(item.get("seat"), -2) == seat]
            vpip = any(
                str(item.get("action", "")) in {"call", "bet", "raise"}
                for item in own_pre
            )
            pfr = any(
                str(item.get("action", "")) in AGGRESSIVE_ACTIONS for item in own_pre
            )
            aggressive = 0
            calls = 0
            faced = 0
            folds = 0
            for round_actions in grouped.values():
                prior_raiser: int | None = None
                for action in round_actions:
                    action_seat = _int(action.get("seat"), -2)
                    action_name = str(action.get("action", ""))
                    if action_seat == seat:
                        aggressive += int(action_name in AGGRESSIVE_ACTIONS)
                        calls += int(action_name == "call")
                        if prior_raiser is not None and prior_raiser != seat:
                            faced += 1
                            folds += int(action_name == "fold")
                    if action_name in AGGRESSIVE_ACTIONS:
                        prior_raiser = action_seat
            additions.append(
                (
                    name,
                    (match_id, leg, hand_number),
                    OpponentHandObservation(vpip, pfr, aggressive, calls, faced, folds),
                )
            )
    if additions:
        with _RULE_MEMORY_LOCK:
            for name, key, observation in additions:
                memory = _OPPONENT_MEMORY.setdefault(name, {})
                memory[key] = observation
                while len(memory) > 500:
                    memory.pop(next(iter(memory)))


def _profile_for_name(name: str, busted: bool) -> OpponentProfile:
    with _RULE_MEMORY_LOCK:
        observations = tuple(_OPPONENT_MEMORY.get(name, {}).values())
    hands = len(observations)
    vpip = (sum(item.vpip for item in observations) + 2) / (hands + 4)
    pfr = (sum(item.pfr for item in observations) + 1) / (hands + 4)
    aggressive = sum(item.aggressive_actions for item in observations)
    calls = sum(item.calls for item in observations)
    faced = sum(item.faced_raise for item in observations)
    folds = sum(item.folded_to_raise for item in observations)
    aggression_factor = (aggressive + 1) / (calls + 1)
    aggression_rate = (aggressive + 3) / (aggressive + calls + 10)
    fold_to_raise = (folds + 2) / (faced + 6)
    named_prior = {
        "Theo": "maniac",
        "Bram": "maniac",
        "Miles": "lag",
        "Dana": "tag",
        "Rhea": "tag",
    }.get(name)
    if hands >= 8 and aggression_factor >= 2.7 and pfr >= 0.34:
        archetype = "maniac"
    elif hands >= 8 and vpip >= 0.58 and pfr >= 0.27:
        archetype = "lag"
    elif hands >= 8 and vpip <= 0.42 and pfr >= 0.16:
        archetype = "tag"
    else:
        archetype = named_prior or "unknown"
    return OpponentProfile(
        fold_to_raise,
        aggression_rate,
        hands,
        name,
        vpip,
        pfr,
        aggression_factor,
        fold_to_raise,
        busted,
        archetype,
    )


def _last_aggressor_seat(state: Mapping[str, Any]) -> int | None:
    your_seat = _int(state.get("your_seat"), -1)
    round_name = str(state.get("round", ""))
    for action in reversed(_actions(state.get("current_hand_actions"))):
        seat = _int(action.get("seat"), -2)
        if (
            seat != your_seat
            and str(action.get("round", "")) == round_name
            and str(action.get("action", "")) in AGGRESSIVE_ACTIONS
        ):
            return seat
    return None


def _leaderboard(state: Mapping[str, Any]) -> tuple[int, int, int | None]:
    your_seat = _int(state.get("your_seat"), -1)
    own_delta = 0
    opponents: list[tuple[int, int]] = []
    players = state.get("players")
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        for player in players:
            if not isinstance(player, Mapping):
                continue
            seat = _int(player.get("seat"), -2)
            delta = _int(player.get("chip_delta"), 0)
            if seat == your_seat:
                own_delta = delta
            else:
                opponents.append((delta, seat))
    if not opponents:
        return own_delta, -200, None
    leader_delta, leader_seat = max(opponents)
    return own_delta, leader_delta, leader_seat


def _leader_is_exposed(
    state: Mapping[str, Any],
    leader_seat: int | None,
    aggressor_seat: int | None,
) -> bool:
    """Whether the current pot offers a direct chance to take the leader's chips."""

    if leader_seat is None:
        return False
    if aggressor_seat == leader_seat:
        return True
    for action in reversed(_actions(state.get("current_hand_actions"))):
        if _int(action.get("seat"), -2) != leader_seat:
            continue
        return str(action.get("action", "")) in {"call", "bet", "raise"}
    players = state.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return False
    for player in players:
        if not isinstance(player, Mapping):
            continue
        if _int(player.get("seat"), -2) != leader_seat:
            continue
        return (
            player.get("folded") is not True
            and player.get("busted") is not True
            and player.get("all_in") is True
        )
    return False


def _strict_majority_locked(
    state: Mapping[str, Any], live_stack: int, own_delta: int
) -> bool:
    """A strict chip majority cannot be overtaken even if opponents consolidate."""

    if own_delta < 10:
        return False
    players = state.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return False
    seats = sum(isinstance(player, Mapping) for player in players)
    if seats < 2:
        return own_delta >= 10
    starting_stack = max(1, _int(state.get("starting_stack"), 200))
    total_chips = seats * starting_stack
    return live_stack * 2 > total_chips


def _final_hand_rank_locked(
    state: Mapping[str, Any], live_stack: int, own_delta: int
) -> bool:
    """Whether folding/checking guarantees rank one after the current final hand."""

    hand = _int(state.get("hand_number"), 0)
    total_hands = _int(state.get("total_hands"), 0)
    if hand < 1 or hand != total_hands or own_delta < 10:
        return False
    your_seat = _int(state.get("your_seat"), -1)
    pot = max(0, _int(state.get("pot"), 0))
    players = state.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return False
    worst_opponent = 0
    for player in players:
        if not isinstance(player, Mapping):
            continue
        if _int(player.get("seat"), -2) == your_seat:
            continue
        opponent_stack = max(0, _int(player.get("stack"), 0))
        can_win_pot = (
            player.get("folded") is not True and player.get("busted") is not True
        )
        worst_opponent = max(
            worst_opponent,
            opponent_stack + (pot if can_win_pot else 0),
        )
    return live_stack > worst_opponent


def _has_all_in_opponent(state: Mapping[str, Any]) -> bool:
    your_seat = _int(state.get("your_seat"), -1)
    players = state.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return False
    return any(
        isinstance(player, Mapping)
        and _int(player.get("seat"), -2) != your_seat
        and player.get("folded") is not True
        and player.get("busted") is not True
        and player.get("all_in") is True
        for player in players
    )


def _showdown_call_floor(
    profile: OpponentProfile,
    aggression_count: int,
    risk: float,
    opponents: int,
    leader_opportunity: bool,
) -> float:
    """Minimum rule percentile needed against the observed betting range."""

    base = {
        "maniac": 0.63,
        "lag": 0.70,
        "tag": 0.82,
        "mixed": 0.75,
        "unknown": 0.76,
    }.get(profile.archetype, 0.76)
    base += 0.055 * max(0, aggression_count - 1)
    base += 0.045 if risk >= 0.35 else 0.02 if risk >= 0.18 else 0.0
    base += 0.025 if opponents >= 3 else 0.0
    if leader_opportunity:
        base -= 0.035
    return _clamp(base, 0.58, 0.93)


def _is_late_position(state: Mapping[str, Any], round_name: str) -> bool:
    your_seat = _int(state.get("your_seat"), -1)
    button = _int(state.get("button_seat"), -2)
    if your_seat < 0 or button < 0:
        return False
    players = state.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return False
    seated: list[int] = []
    acting: set[int] = set()
    for player in players:
        if not isinstance(player, Mapping) or player.get("busted") is True:
            continue
        seat = _int(player.get("seat"), -1)
        if seat < 0:
            continue
        seated.append(seat)
        if player.get("folded") is not True and player.get("all_in") is not True:
            acting.add(seat)
    seated.sort()
    if not seated or button not in seated or your_seat not in acting:
        return False

    button_index = seated.index(button)
    if round_name == "post_reveal":
        start_index = (button_index + 1) % len(seated)
        late_count = 1
    else:
        # The first pre-reveal actor sits immediately after the big blind, the
        # second live seat past the button. This also handles busted-seat skips.
        start_index = (button_index + 3) % len(seated)
        late_count = 2
    order = seated[start_index:] + seated[:start_index]
    remaining_order = [seat for seat in order if seat in acting]
    return your_seat in set(remaining_order[-late_count:])


def _aggressive_move(
    state: Mapping[str, Any],
    legal: set[Action],
    own_bet: int,
    to_call: int,
    pot: int,
    pot_fraction: float,
    *,
    all_in: bool = False,
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
    desired = (
        maximum if all_in else own_bet + to_call + max(1, round(pot * pot_fraction))
    )
    if phase == 2 and not all_in:
        # Phase 2 is scored on a fixed threshold, so retain its conservative cap.
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
        _OPPONENT_MEMORY.clear()


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