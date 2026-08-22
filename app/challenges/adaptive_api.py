"""Adaptive API Gateway challenge routes."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError


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


class DecodedPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    adaptInput: AdaptInput


class AdaptOutput(BaseModel):
    id: str
    name: str
    action: str
    priority: int


class SolveResponse(BaseModel):
    adaptOutput: AdaptOutput


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

    return SolveResponse(
        adaptOutput=AdaptOutput(
            id=source.user.id,
            name=source.user.fullName,
            action=source.action.lower(),
            priority=PRIORITY_VALUES[source.metadata.priority],
        )
    )


# Keep /solve for the already-registered competition callback. The namespaced
# route is the stable convention for this repository going forward.
router.add_api_route("/solve", solve_payload, methods=["POST"], response_model=SolveResponse)
router.add_api_route(
    "/adaptive-api/solve",
    solve_payload,
    methods=["POST"],
    response_model=SolveResponse,
)
