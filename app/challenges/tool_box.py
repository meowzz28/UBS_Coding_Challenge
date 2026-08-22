"""MCP tools for Tool Box Phase 1."""

from __future__ import annotations

import base64
import binascii
import math
import statistics
from io import BytesIO
from typing import Annotated, Literal

from fastapi import APIRouter
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from PIL import Image, UnidentifiedImageError
from pydantic import Field


ASSISTANT_NAME = "Nova Box"
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048

server = MCPServer(
    name="UBS Tool Box",
    title="Tool Box Phase 1",
    description="Basic tools for identity, arithmetic, and PNG shape recognition.",
    instructions=(
        "Use get_name when asked for your name, calculate for arithmetic, and "
        "identify_shape for base64-encoded PNG images. Combine these tools when "
        "a request contains more than one kind of task. Return the tool results "
        "directly and do not guess."
    ),
    version="1.0.0",
)


@server.tool()
def get_name() -> str:
    """Return your assigned name as a plain string."""

    return ASSISTANT_NAME


@server.tool()
def calculate(
    left: Annotated[int, Field(ge=-100, le=100, description="Left integer operand")],
    operator: Annotated[
        Literal["+", "-", "*", "/"],
        Field(description="One arithmetic operator: +, -, *, or /"),
    ],
    right: Annotated[int, Field(ge=-100, le=100, description="Right integer operand")],
) -> int | float:
    """Calculate one binary arithmetic expression using integers from -100 to 100."""

    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0:
        raise ValueError("division by zero is undefined")
    return left / right


@server.tool()
def identify_shape(
    image_base64: Annotated[
        str,
        Field(description="A base64-encoded PNG image, optionally as a data URI"),
    ],
) -> Literal["rectangle", "triangle", "circle"]:
    """Identify the single prominent shape in a base64 PNG image."""

    image = _decode_png(image_base64)
    mask = _foreground_mask(image)
    return _classify_mask(mask)


def _decode_png(encoded: str) -> Image.Image:
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or "image/png" not in header.lower():
            raise ValueError("image must be a base64-encoded PNG")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 must contain valid Base64") from exc

    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("PNG must be between 1 byte and 2 MB")

    try:
        source = Image.open(BytesIO(raw))
        if source.format != "PNG":
            raise ValueError("decoded image must be a PNG")
        if max(source.size) > MAX_IMAGE_DIMENSION or min(source.size) < 8:
            raise ValueError("PNG dimensions must be between 8 and 2048 pixels")
        source.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("decoded image must be a valid PNG") from exc

    rgba = source.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def _foreground_mask(image: Image.Image) -> list[list[bool]]:
    width, height = image.size
    pixels = image.load()
    border = [pixels[x, 0] for x in range(width)]
    border += [pixels[x, height - 1] for x in range(width)]
    border += [pixels[0, y] for y in range(1, height - 1)]
    border += [pixels[width - 1, y] for y in range(1, height - 1)]
    background = tuple(
        int(statistics.median(pixel[channel] for pixel in border))
        for channel in range(3)
    )

    threshold = 36
    mask = [
        [
            math.sqrt(
                sum(
                    (pixels[x, y][channel] - background[channel]) ** 2
                    for channel in range(3)
                )
            )
            >= threshold
            for x in range(width)
        ]
        for y in range(height)
    ]

    foreground_count = sum(sum(row) for row in mask)
    if foreground_count < max(12, (width * height) // 1000):
        raise ValueError("PNG does not contain a prominent shape")
    return mask


def _classify_mask(
    mask: list[list[bool]],
) -> Literal["rectangle", "triangle", "circle"]:
    points = [(x, y) for y, row in enumerate(mask) for x, value in enumerate(row) if value]
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1

    if min(box_width, box_height) < 5:
        raise ValueError("detected shape is too small to classify")

    cropped = [row[min_x : max_x + 1] for row in mask[min_y : max_y + 1]]
    row_spans = _span_profile(cropped)
    column_spans = _span_profile(list(map(list, zip(*cropped))))
    row_flatness, row_end_difference = _profile_features(row_spans)
    column_flatness, column_end_difference = _profile_features(column_spans)

    # Rectangles keep nearly the same outer span along both axes, whether the
    # shape is filled or outlined.
    if row_flatness >= 0.72 and column_flatness >= 0.72:
        return "rectangle"

    # A triangle has an apex: on at least one axis its span changes strongly
    # from one end of the bounding box to the other. This also handles a rotated
    # left- or right-facing triangle by considering the column profile.
    if max(row_end_difference, column_end_difference) >= 0.34:
        return "triangle"

    # A circle narrows at both ends and reaches its largest span near the middle.
    return "circle"


def _span_profile(rows: list[list[bool]]) -> list[int]:
    spans: list[int] = []
    for row in rows:
        positions = [index for index, value in enumerate(row) if value]
        spans.append(positions[-1] - positions[0] + 1 if positions else 0)
    return spans


def _profile_features(spans: list[int]) -> tuple[float, float]:
    maximum = max(spans)
    normalized = [span / maximum for span in spans]
    end_size = max(1, len(normalized) // 6)
    first = statistics.median(normalized[:end_size])
    last = statistics.median(normalized[-end_size:])
    flatness = sum(value >= 0.82 for value in normalized) / len(normalized)
    return flatness, abs(first - last)


router = APIRouter(prefix="/tool-box", tags=["tool-box"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Render terminates TLS and controls the Host header before forwarding to this
# process, so disabling the SDK's localhost-only DNS-rebinding default is the
# appropriate reverse-proxy configuration.
http_app = server.streamable_http_app(
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
