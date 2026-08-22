"""Stateful Phase 1 and Phase 2 implementation for Ghost Chains."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

LOOKBACK = timedelta(hours=24)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    txId: Annotated[str, Field(min_length=1)]
    fromUserId: Annotated[str, Field(min_length=1)]
    toUserId: Annotated[str, Field(min_length=1)]
    amount: float
    createdAt: datetime
    ipAddress: str | None = None
    deviceId: str | None = None

    @field_validator("amount")
    @classmethod
    def amount_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("amount must be finite")
        return value

    @field_validator("createdAt")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("createdAt must include a timezone")
        return value.astimezone(timezone.utc)


class TransactionsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transactions: list[Transaction]


class TransactionResult(BaseModel):
    txId: str
    riskScore: float


class TransactionsResponse(BaseModel):
    transactions: list[TransactionResult]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clearTransactions: bool


@dataclass(frozen=True)
class StoredTransaction:
    transaction: Transaction
    canonical_payload: dict[str, Any]
    score: float


@dataclass(frozen=True)
class IdentityDimensionSignal:
    """Evidence contributed by one independent identity dimension."""

    agreement: float = 0.0
    divergence: float = 0.0
    missing: float = 0.0
    disconnected_reuse: float = 0.0


class DuplicateTransactionConflict(Exception):
    """Raised when a txId is reused with a different payload."""


class TransactionGraph:
    """Rolling directed multigraph with structural and identity evidence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._outgoing: dict[str, Counter[str]] = defaultdict(Counter)
            self._incoming: dict[str, Counter[str]] = defaultdict(Counter)
            self._active_transactions: dict[str, StoredTransaction] = {}
            self._idempotency: dict[str, StoredTransaction] = {}
            self._incoming_tx_ids: dict[str, set[str]] = defaultdict(set)
            self._outgoing_tx_ids: dict[str, set[str]] = defaultdict(set)
            self._identity_tx_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
            self._expiry_heap: list[tuple[datetime, int, str]] = []
            self._watermark: datetime | None = None
            self._sequence = 0

    def process_batch(self, transactions: list[Transaction]) -> list[TransactionResult]:
        with self._lock:
            self._preflight_batch(transactions)
            return [self._process_one(transaction) for transaction in transactions]

    def _preflight_batch(self, transactions: list[Transaction]) -> None:
        """Reject conflicting identifiers before mutating any batch state."""

        payloads: dict[str, dict[str, Any]] = {}
        for transaction in transactions:
            canonical = self._canonical_payload(transaction)
            previous_payload = payloads.get(transaction.txId)
            if previous_payload is not None and previous_payload != canonical:
                raise DuplicateTransactionConflict(transaction.txId)
            payloads[transaction.txId] = canonical

            existing = self._idempotency.get(transaction.txId)
            if existing is not None and existing.canonical_payload != canonical:
                raise DuplicateTransactionConflict(transaction.txId)

    def _process_one(self, transaction: Transaction) -> TransactionResult:
        canonical = self._canonical_payload(transaction)
        existing = self._idempotency.get(transaction.txId)
        if existing is not None:
            return TransactionResult(txId=transaction.txId, riskScore=existing.score)

        self._advance_window(transaction.createdAt)
        score = round(self._calculate_risk(transaction), 6)
        stored = StoredTransaction(transaction, canonical, score)
        self._idempotency[transaction.txId] = stored

        # A late transaction that is already older than the current window still
        # receives a deterministic score, but it must not re-enter active state.
        if self._is_within_active_window(transaction.createdAt):
            self._add_transaction(stored)
        return TransactionResult(txId=transaction.txId, riskScore=score)

    @staticmethod
    def _canonical_payload(transaction: Transaction) -> dict[str, Any]:
        return transaction.model_dump(mode="json", exclude_none=False)

    def _advance_window(self, timestamp: datetime) -> None:
        if self._watermark is None or timestamp > self._watermark:
            self._watermark = timestamp

        cutoff = self._watermark - LOOKBACK
        while self._expiry_heap and self._expiry_heap[0][0] < cutoff:
            _, _, tx_id = heapq.heappop(self._expiry_heap)
            stored = self._active_transactions.pop(tx_id, None)
            if stored is not None:
                self._remove_transaction(stored)

    def _is_within_active_window(self, timestamp: datetime) -> bool:
        return self._watermark is None or timestamp >= self._watermark - LOOKBACK

    def _calculate_risk(self, transaction: Transaction) -> float:
        source = transaction.fromUserId
        destination = transaction.toUserId
        if source == destination:
            structural = 0.98
        else:
            structural = self._calculate_structural_risk(source, destination)

        identity = self._calculate_identity_risk(transaction, structural)
        combined = 1.0 - (1.0 - structural) * (1.0 - identity)
        return self._clamp(combined)

    def _calculate_structural_risk(self, source: str, destination: str) -> float:

        reverse_distance = self._shortest_distance(destination, source)
        forward_distance = self._shortest_distance(source, destination)
        source_ancestors = self._reachable(source, self._incoming)
        destination_ancestors = self._reachable(destination, self._incoming)
        destination_descendants = self._reachable(destination, self._outgoing)
        common_ancestors = source_ancestors & destination_ancestors

        path_span = (len(source_ancestors) + 1) * (len(destination_descendants) + 1)
        span_bonus = min(0.10, math.log2(path_span + 1) * 0.02)

        # Closing a directed return path creates a cycle. Existing cycles at the
        # destination make an additional independent return route more suspicious.
        if reverse_distance is not None:
            score = 0.64 + (0.08 / max(reverse_distance, 1))
            if self._node_is_in_cycle(destination):
                score += 0.14
            score += min(0.075, self._degree(destination, self._incoming) * 0.025)
            score += min(0.045, self._degree(source, self._outgoing) * 0.015)
            return min(0.98, score + (span_bonus / 2))

        # A direct edge that replaces a longer existing route shortens a path;
        # an already-direct repeated pair adds much less new structural signal.
        if forward_distance is not None:
            if forward_distance == 1:
                return min(0.16, 0.07 + (span_bonus / 2))
            return min(
                0.45, 0.22 + min(0.12, (forward_distance - 1) * 0.04) + span_bonus
            )

        # Two branches with a shared upstream ancestor converging on the same
        # destination create multiple routes through the graph.
        if common_ancestors:
            return min(
                0.58, 0.30 + min(0.18, len(common_ancestors) * 0.06) + span_bonus
            )

        source_seen = self._node_is_known(source)
        destination_seen = self._node_is_known(destination)
        if not source_seen and not destination_seen:
            return 0.02
        if source_seen and not destination_seen:
            return min(0.24, 0.10 + span_bonus)
        if not source_seen and destination_seen:
            return min(0.30, 0.15 + span_bonus)
        return min(0.34, 0.18 + span_bonus)

    def _calculate_identity_risk(
        self,
        transaction: Transaction,
        structural_risk: float,
    ) -> float:
        """Combine IP and device evidence without treating absence globally."""

        signals = [
            self._identity_dimension_signal(transaction, "ipAddress"),
            self._identity_dimension_signal(transaction, "deviceId"),
        ]
        dimension_risks: list[float] = []
        agreements: list[float] = []
        for signal in signals:
            # Divergence and missingness matter only when the transaction extends
            # a flow that previously carried the attribute. Disconnected reuse is
            # deliberately weaker: shared Wi-Fi or NAT can be legitimate.
            anomaly = (
                0.34 * signal.divergence
                + 0.29 * signal.missing
                + 0.15 * signal.disconnected_reuse
            )
            dimension_risks.append(self._clamp(anomaly))
            agreements.append(signal.agreement)

        independent_risk = 1.0
        for dimension_risk in dimension_risks:
            independent_risk *= 1.0 - dimension_risk
        independent_risk = 1.0 - independent_risk

        # Identity anomalies become more compelling as the structural flow grows.
        # A disconnected shared identity still contributes a small coordination
        # hint through the non-zero baseline coupling.
        coupling = 0.34 + 0.66 * structural_risk
        risk = independent_risk * coupling

        # Consistent identity is not anomalous, but on an established structural
        # path it modestly reinforces the common-control interpretation.
        agreement = sum(agreements) / len(agreements)
        risk += min(0.035, structural_risk * agreement * 0.035)
        return self._clamp(risk)

    def _identity_dimension_signal(
        self,
        transaction: Transaction,
        attribute: str,
    ) -> IdentityDimensionSignal:
        source = transaction.fromUserId
        destination = transaction.toUserId
        current = self._identity_value(getattr(transaction, attribute))

        direct_values = self._identity_values(
            self._incoming_tx_ids.get(source, set()), attribute
        )
        upstream_values = self._identity_values(
            self._upstream_tx_ids(source), attribute
        )
        sibling_values = self._identity_values(
            self._outgoing_tx_ids.get(source, set()),
            attribute,
        )

        agreement = 0.0
        divergence = 0.0
        missing = 0.0
        if upstream_values:
            if current is None:
                missing = 1.0
            else:
                upstream_total = sum(upstream_values.values())
                upstream_agreement = upstream_values[current] / upstream_total
                if direct_values:
                    direct_total = sum(direct_values.values())
                    direct_agreement = direct_values[current] / direct_total
                    agreement = 0.72 * direct_agreement + 0.28 * upstream_agreement
                else:
                    agreement = upstream_agreement
                divergence = 1.0 - agreement
        elif sibling_values:
            # A branch changing identity is meaningful, but less direct than an
            # identity change between consecutive legs of a flow.
            if current is None:
                missing = 0.55
            else:
                total = sum(sibling_values.values())
                agreement = sibling_values[current] / total
                divergence = 0.55 * (1.0 - agreement)

        disconnected_reuse = 0.0
        if current is not None:
            disconnected_components = self._disconnected_identity_components(
                attribute,
                current,
                source,
                destination,
            )
            disconnected_reuse = min(
                1.0,
                math.log2(disconnected_components + 1) / 2.0,
            )

        return IdentityDimensionSignal(
            agreement=agreement,
            divergence=divergence,
            missing=missing,
            disconnected_reuse=disconnected_reuse,
        )

    def _identity_values(
        self,
        tx_ids: set[str],
        attribute: str,
    ) -> Counter[str]:
        values: Counter[str] = Counter()
        for tx_id in tx_ids:
            stored = self._active_transactions.get(tx_id)
            if stored is None:
                continue
            value = self._identity_value(getattr(stored.transaction, attribute))
            if value is not None:
                values[value] += 1
        return values

    def _upstream_tx_ids(self, node: str) -> set[str]:
        """Return active transaction legs on every directed path into a node."""

        tx_ids: set[str] = set()
        visited = {node}
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for tx_id in self._incoming_tx_ids.get(current, set()):
                if tx_id in tx_ids:
                    continue
                tx_ids.add(tx_id)
                stored = self._active_transactions.get(tx_id)
                if stored is None:
                    continue
                predecessor = stored.transaction.fromUserId
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)
        return tx_ids

    def _disconnected_identity_components(
        self,
        attribute: str,
        value: str,
        source: str,
        destination: str,
    ) -> int:
        current_component = self._weak_component(source) | self._weak_component(
            destination
        )
        current_component.update((source, destination))
        disconnected: set[frozenset[str]] = set()
        for tx_id in self._identity_tx_ids.get((attribute, value), set()):
            stored = self._active_transactions.get(tx_id)
            if stored is None:
                continue
            previous = stored.transaction
            if (
                previous.fromUserId in current_component
                or previous.toUserId in current_component
            ):
                continue
            component = self._weak_component(previous.fromUserId)
            component.update((previous.fromUserId, previous.toUserId))
            disconnected.add(frozenset(component))
        return len(disconnected)

    def _weak_component(self, node: str) -> set[str]:
        visited = {node}
        queue = deque([node])
        while queue:
            current = queue.popleft()
            neighbors = set(self._outgoing.get(current, {})) | set(
                self._incoming.get(current, {})
            )
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    @staticmethod
    def _identity_value(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _node_is_known(self, node: str) -> bool:
        return bool(self._outgoing.get(node) or self._incoming.get(node))

    def _node_is_in_cycle(self, node: str) -> bool:
        for neighbor in self._outgoing.get(node, {}):
            if neighbor == node or self._shortest_distance(neighbor, node) is not None:
                return True
        return False

    def _shortest_distance(self, start: str, target: str) -> int | None:
        if start == target:
            return 0

        queue: deque[tuple[str, int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            node, distance = queue.popleft()
            for neighbor in self._outgoing.get(node, {}):
                if neighbor == target:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    @staticmethod
    def _reachable(node: str, graph: dict[str, Counter[str]]) -> set[str]:
        visited = {node}
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        visited.remove(node)
        return visited

    @staticmethod
    def _degree(node: str, graph: dict[str, Counter[str]]) -> int:
        return sum(graph.get(node, {}).values())

    def _add_transaction(self, stored: StoredTransaction) -> None:
        transaction = stored.transaction
        self._active_transactions[transaction.txId] = stored
        self._add_edge(transaction.fromUserId, transaction.toUserId)
        self._outgoing_tx_ids[transaction.fromUserId].add(transaction.txId)
        self._incoming_tx_ids[transaction.toUserId].add(transaction.txId)
        for attribute in ("ipAddress", "deviceId"):
            value = self._identity_value(getattr(transaction, attribute))
            if value is not None:
                self._identity_tx_ids[(attribute, value)].add(transaction.txId)

        self._sequence += 1
        heapq.heappush(
            self._expiry_heap,
            (transaction.createdAt, self._sequence, transaction.txId),
        )

    def _remove_transaction(self, stored: StoredTransaction) -> None:
        transaction = stored.transaction
        self._remove_edge(transaction.fromUserId, transaction.toUserId)
        self._discard_index(
            self._outgoing_tx_ids,
            transaction.fromUserId,
            transaction.txId,
        )
        self._discard_index(
            self._incoming_tx_ids,
            transaction.toUserId,
            transaction.txId,
        )
        for attribute in ("ipAddress", "deviceId"):
            value = self._identity_value(getattr(transaction, attribute))
            if value is not None:
                self._discard_index(
                    self._identity_tx_ids,
                    (attribute, value),
                    transaction.txId,
                )

    def _add_edge(self, source: str, destination: str) -> None:
        self._outgoing[source][destination] += 1
        self._incoming[destination][source] += 1

    def _remove_edge(self, source: str, destination: str) -> None:
        self._decrement(self._outgoing, source, destination)
        self._decrement(self._incoming, destination, source)

    @staticmethod
    def _decrement(graph: dict[str, Counter[str]], node: str, neighbor: str) -> None:
        neighbors = graph.get(node)
        if neighbors is None:
            return
        neighbors[neighbor] -= 1
        if neighbors[neighbor] <= 0:
            del neighbors[neighbor]
        if not neighbors:
            del graph[node]

    @staticmethod
    def _discard_index(
        index: dict[Any, set[str]],
        key: Any,
        tx_id: str,
    ) -> None:
        values = index.get(key)
        if values is None:
            return
        values.discard(tx_id)
        if not values:
            del index[key]


graph = TransactionGraph()
router = APIRouter(prefix="/ghost-chains", tags=["ghost-chains"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/reset", response_model=ResetRequest)
def reset(request: ResetRequest) -> ResetRequest:
    if request.clearTransactions:
        graph.reset()
    return request


@router.post("/transactions", response_model=TransactionsResponse)
def process_transactions(request: TransactionsRequest) -> TransactionsResponse:
    try:
        results = graph.process_batch(request.transactions)
    except DuplicateTransactionConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=f"txId '{exc.args[0]}' was already used with a different payload",
        ) from exc
    return TransactionsResponse(transactions=results)
