"""Adaptive API Gateway challenge routes."""

from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: str


class User(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    fullName: str


class Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    priority: Literal["LOW", "MEDIUM", "HIGH"]


class AdaptInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user: User
    action: str
    metadata: Metadata


class Heartbeat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    service: str
    timestamp: int
    latencyMs: int = Field(ge=0)
    status: Literal["OK", "FAIL"]


class SloQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    service: str
    since: int


class DecodedPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    adaptInput: AdaptInput
    heartbeats: list[Heartbeat] | None = None
    sloQuery: SloQuery | None = None

    @model_validator(mode="after")
    def require_complete_slo_input(self) -> DecodedPayload:
        if (self.heartbeats is None) != (self.sloQuery is None):
            raise ValueError("heartbeats and sloQuery must be provided together")
        return self


class AdaptOutput(BaseModel):
    id: str
    name: str
    action: str
    priority: int


class SloOutput(BaseModel):
    availability: float
    p95LatencyMs: int


class SolveResponse(BaseModel):
    adaptOutput: AdaptOutput
    sloOutput: SloOutput | None = None


PRIORITY_VALUES = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

router = APIRouter(tags=["adaptive-api"])


def decode_payload(encoded_payload: str) -> DecodedPayload:
    try:
        decoded_bytes = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="payload must be valid Base64") from exc

    try:
        decoded_json: Any = json.loads(decoded_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="decoded payload must be valid UTF-8 JSON",
        ) from exc

    try:
        return DecodedPayload.model_validate(decoded_json)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="decoded payload does not match the expected V1 model",
        ) from exc


def solve_payload(request: SolveRequest) -> SolveResponse:
    decoded = decode_payload(request.payload)
    source = decoded.adaptInput

    response = SolveResponse(
        adaptOutput=AdaptOutput(
            id=source.user.id,
            name=source.user.fullName,
            action=source.action.lower(),
            priority=PRIORITY_VALUES[source.metadata.priority],
        )
    )
    if decoded.heartbeats is not None and decoded.sloQuery is not None:
        response.sloOutput = calculate_slo(decoded.heartbeats, decoded.sloQuery)
    return response


def calculate_slo(heartbeats: list[Heartbeat], query: SloQuery) -> SloOutput:
    """Calculate availability and nearest-rank p95 for the requested time window."""

    matching = [
        heartbeat
        for heartbeat in heartbeats
        if heartbeat.service == query.service and heartbeat.timestamp >= query.since
    ]
    if not matching:
        return SloOutput(availability=0.0, p95LatencyMs=0)

    available = sum(heartbeat.status == "OK" for heartbeat in matching)
    latencies = sorted(heartbeat.latencyMs for heartbeat in matching)
    percentile_index = math.ceil(0.95 * len(latencies)) - 1
    return SloOutput(
        availability=available / len(matching),
        p95LatencyMs=latencies[percentile_index],
    )


# Keep /solve for the already-registered competition callback. The namespaced
# route is the stable convention for this repository going forward.
router.add_api_route(
    "/solve",
    solve_payload,
    methods=["POST"],
    response_model=SolveResponse,
    response_model_exclude_none=True,
)
router.add_api_route(
    "/adaptive-api/solve",
    solve_payload,
    methods=["POST"],
    response_model=SolveResponse,
    response_model_exclude_none=True,
)
