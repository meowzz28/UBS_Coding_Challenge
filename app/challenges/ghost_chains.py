"""Stateful Phase 1 implementation for the Ghost Chains challenge."""

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


class DuplicateTransactionConflict(Exception):
    """Raised when a txId is reused with a different payload."""


class TransactionGraph:
    """In-memory, rolling directed multigraph for Phase 1 scoring."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._outgoing: dict[str, Counter[str]] = defaultdict(Counter)
            self._incoming: dict[str, Counter[str]] = defaultdict(Counter)
            self._transactions: dict[str, StoredTransaction] = {}
            self._expiry_heap: list[tuple[datetime, int, str]] = []
            self._watermark: datetime | None = None
            self._sequence = 0

    def process_batch(self, transactions: list[Transaction]) -> list[TransactionResult]:
        with self._lock:
            return [self._process_one(transaction) for transaction in transactions]

    def _process_one(self, transaction: Transaction) -> TransactionResult:
        canonical = transaction.model_dump(mode="json", exclude_none=False)
        existing = self._transactions.get(transaction.txId)
        if existing is not None:
            if existing.canonical_payload != canonical:
                raise DuplicateTransactionConflict(transaction.txId)
            return TransactionResult(txId=transaction.txId, riskScore=existing.score)

        self._advance_window(transaction.createdAt)
        score = round(self._calculate_risk(transaction.fromUserId, transaction.toUserId), 6)
        stored = StoredTransaction(transaction, canonical, score)
        self._transactions[transaction.txId] = stored
        self._add_edge(transaction.fromUserId, transaction.toUserId)

        self._sequence += 1
        heapq.heappush(
            self._expiry_heap,
            (transaction.createdAt, self._sequence, transaction.txId),
        )
        return TransactionResult(txId=transaction.txId, riskScore=score)

    def _advance_window(self, timestamp: datetime) -> None:
        if self._watermark is None or timestamp > self._watermark:
            self._watermark = timestamp

        cutoff = self._watermark - LOOKBACK
        while self._expiry_heap and self._expiry_heap[0][0] < cutoff:
            _, _, tx_id = heapq.heappop(self._expiry_heap)
            stored = self._transactions.pop(tx_id, None)
            if stored is not None:
                self._remove_edge(
                    stored.transaction.fromUserId,
                    stored.transaction.toUserId,
                )

    def _calculate_risk(self, source: str, destination: str) -> float:
        if source == destination:
            return 0.98

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
            return min(0.45, 0.22 + min(0.12, (forward_distance - 1) * 0.04) + span_bonus)

        # Two branches with a shared upstream ancestor converging on the same
        # destination create multiple routes through the graph.
        if common_ancestors:
            return min(0.58, 0.30 + min(0.18, len(common_ancestors) * 0.06) + span_bonus)

        source_seen = self._node_is_known(source)
        destination_seen = self._node_is_known(destination)
        if not source_seen and not destination_seen:
            return 0.02
        if source_seen and not destination_seen:
            return min(0.24, 0.10 + span_bonus)
        if not source_seen and destination_seen:
            return min(0.30, 0.15 + span_bonus)
        return min(0.34, 0.18 + span_bonus)

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
