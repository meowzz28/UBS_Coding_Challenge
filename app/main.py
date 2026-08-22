"""FastAPI server for the UBS Adaptive API Gateway challenge."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError


class SolveRequest(BaseModel):
    """Outer request sent by the competition server."""

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

app = FastAPI(
    title="UBS Adaptive API Gateway",
    description="Adapts Base64-encoded V1 requests to the V2 response model.",
    version="1.0.0",
)


def decode_payload(encoded_payload: str) -> DecodedPayload:
    """Decode and validate the Base64-encoded JSON challenge payload."""

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


@app.get("/", tags=["service"])
def service_info() -> dict[str, str]:
    return {
        "service": "UBS Adaptive API Gateway",
        "status": "ready",
        "solveEndpoint": "/solve",
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse, tags=["challenge"])
def solve(request: SolveRequest) -> SolveResponse:
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
