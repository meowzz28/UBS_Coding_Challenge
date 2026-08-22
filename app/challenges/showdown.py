"""SHOWDOWN Phase 1 betting strategy and HTTP routes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from fastapi import APIRouter


Action = Literal["check", "call", "bet", "raise", "fold"]
AGGRESSIVE_ACTIONS = {"bet", "raise"}


class Move(TypedDict, total=False):
    action: Action
    amount: int


@dataclass(frozen=True)
class OpponentProfile:
    """Smoothed tendencies reconstructed from the rolling public history."""

    fold_to_aggression: float
    aggression_rate: float
    observations: int


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
    """Choose a Phase 1 move using equity, price, risk, and opponent history."""

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

    # Phase 1 is threshold-scored rather than proportional: once the live stack
    # can absorb every remaining forced bet and still finish at +10, eliminate
    # voluntary variance. Checking is free; fold only when chips are demanded.
    if _phase_one_target_locked(state, stack):
        if to_call > 0:
            return _first_legal(legal, "fold", "call", "check")
        return _first_legal(legal, "check", "call", "fold")

    if round_name == "post_reveal" and community is not None:
        return _post_reveal_move(
            state=state,
            legal=legal,
            number=number,
            community=community,
            to_call=to_call,
            pot=pot,
            stack=stack,
            own_bet=own_bet,
            profile=profile,
        )

    return _pre_reveal_move(
        state=state,
        legal=legal,
        number=number,
        to_call=to_call,
        pot=pot,
        stack=stack,
        own_bet=own_bet,
        profile=profile,
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


def _post_reveal_move(
    *,
    state: Mapping[str, Any],
    legal: set[Action],
    number: int,
    community: int,
    to_call: int,
    pot: int,
    stack: int,
    own_bet: int,
    profile: OpponentProfile,
) -> Move:
    equity = showdown_equity(number, community)
    is_pair = number == community
    facing_bet = to_call > 0

    if facing_bet:
        price = _pot_odds(to_call, pot)
        risk = to_call / max(1, stack)

        # A pair is the nuts in this game. It loses to nothing and only splits
        # against the opponent holding the same number, so build the pot whenever
        # raising remains legal.
        if is_pair:
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

        # Raise premium non-pair hands for value without turning a marginal edge
        # into an unnecessary tournament-sized pot.
        if (
            number >= 12
            and adjusted_equity >= 0.78
            and risk <= 0.30
            and _can_aggress(legal)
        ):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 0.65)

        call_margin = 0.035 if profile.aggression_rate >= 0.45 else 0.065
        affordable = risk <= 0.52 or adjusted_equity >= 0.84
        if "call" in legal and affordable and adjusted_equity >= price + call_margin:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")

    # When checked to, value bet the strongest portion of the range. Occasionally
    # checking a pair against an aggressive opponent preserves an inducing line.
    if is_pair:
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
    number: int,
    to_call: int,
    pot: int,
    stack: int,
    own_bet: int,
    profile: OpponentProfile,
) -> Move:
    equity = pre_reveal_equity(number)
    small_blind = max(1, _int(state.get("small_blind"), 1))
    actions = _actions(state.get("current_hand_actions"))
    blind_completion = to_call <= small_blind and not actions

    # The button's opening decision is a one-chip completion rather than evidence
    # of opponent strength. Continue nearly every hand that is getting the price.
    if blind_completion:
        if number >= 10 and _can_aggress(legal):
            fraction = 0.72 if number >= 12 else 0.52
            return _aggressive_move(state, legal, own_bet, to_call, pot, fraction)
        if number >= 3 and "call" in legal:
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

        if (
            number >= 12
            and risk <= 0.24
            and adjusted_equity >= 0.76
            and _can_aggress(legal)
        ):
            return _aggressive_move(state, legal, own_bet, to_call, pot, 0.62)

        # Avoid calling off most of the match with a hand that has not yet seen the
        # community number. The top number remains profitable against wide shoves.
        affordable = risk <= 0.38 or (number == 13 and adjusted_equity >= 0.84)
        margin = 0.075 if profile.aggression_rate < 0.40 else 0.045
        if "call" in legal and affordable and adjusted_equity >= price + margin:
            return {"action": "call"}
        return _first_legal(legal, "fold", "call", "check")

    if number >= 11 and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.62)
    if number >= 9 and _can_aggress(legal):
        return _aggressive_move(state, legal, own_bet, 0, pot, 0.42)

    if _can_aggress(legal) and number <= 3:
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

                if seat == your_seat and action_name in AGGRESSIVE_ACTIONS:
                    if index + 1 < len(round_actions):
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

    desired = own_bet + to_call + max(1, round(pot * pot_fraction))
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


def _phase_one_target_locked(state: Mapping[str, Any], live_stack: int) -> bool:
    if _int(state.get("phase"), 0) != 1:
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
    return live_stack - future_cost >= starting_stack + 10


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
