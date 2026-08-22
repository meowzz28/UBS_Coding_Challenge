"""MCP tools for Tool Box Phase 1."""

from __future__ import annotations

import ast
import base64
import binascii
import math
import statistics
from collections import deque
from fractions import Fraction
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
MAX_EXPRESSION_LENGTH = 200
MAX_OPERATIONS = 20

server = MCPServer(
    name="UBS Tool Box",
    title="Tool Box Phase 1",
    description="Basic tools for identity, arithmetic, and PNG shape recognition.",
    instructions=(
        "Use get_name when asked for your name. For every arithmetic question, "
        "call calculate exactly once with the entire expression; it applies "
        "standard precedence and parentheses, so never split an expression into "
        "sequential calls or evaluate it left-to-right. Use identify_shape for "
        "base64-encoded PNG images. Combine these tools when a request contains "
        "more than one kind of task. Return tool results directly and do not guess."
    ),
    version="1.1.0",
)


@server.tool()
def get_name() -> str:
    """Return your assigned name as a plain string."""

    return ASSISTANT_NAME


@server.tool()
def calculate(
    expression: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_EXPRESSION_LENGTH,
            description=(
                "The complete arithmetic expression, such as '2 + 3 * 5'. "
                "Pass all operators in one call; standard precedence is automatic."
            ),
        ),
    ],
) -> int | float:
    """Evaluate a complete arithmetic expression with standard operator precedence."""

    normalized = expression.strip().rstrip("?").strip()
    lowered = normalized.lower()
    for prefix in ("what is ", "calculate ", "compute "):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    normalized = normalized.replace("×", "*").replace("÷", "/").replace("−", "-")

    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression must contain valid arithmetic") from exc

    value = _evaluate_expression_node(parsed.body, [0])
    if value.denominator == 1:
        return value.numerator
    return float(value)


def _evaluate_expression_node(node: ast.AST, operation_count: list[int]) -> Fraction:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("operands must be integers")
        if not -100 <= node.value <= 100:
            raise ValueError("each integer operand must be between -100 and 100")
        return Fraction(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression_node(node.operand, operation_count)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div),
    ):
        operation_count[0] += 1
        if operation_count[0] > MAX_OPERATIONS:
            raise ValueError(
                f"expression cannot contain more than {MAX_OPERATIONS} operations"
            )

        left = _evaluate_expression_node(node.left, operation_count)
        right = _evaluate_expression_node(node.right, operation_count)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("division by zero is undefined")
        return left / right

    raise ValueError("only integers, parentheses, and +, -, *, / are supported")


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

    mask, foreground_count = _largest_connected_component(mask)
    if foreground_count < max(12, (width * height) // 1000):
        raise ValueError("PNG does not contain a prominent shape")
    return mask


def _largest_connected_component(
    mask: list[list[bool]],
) -> tuple[list[list[bool]], int]:
    """Discard isolated compression noise and retain the prominent shape."""

    height = len(mask)
    width = len(mask[0])
    visited = bytearray(width * height)
    largest: list[int] = []

    for y in range(height):
        for x in range(width):
            start = y * width + x
            if visited[start] or not mask[y][x]:
                continue

            visited[start] = 1
            queue = deque([start])
            component: list[int] = []
            while queue:
                index = queue.popleft()
                component.append(index)
                current_y, current_x = divmod(index, width)
                for adjacent_y in range(
                    max(0, current_y - 1), min(height, current_y + 2)
                ):
                    for adjacent_x in range(
                        max(0, current_x - 1), min(width, current_x + 2)
                    ):
                        adjacent = adjacent_y * width + adjacent_x
                        if not visited[adjacent] and mask[adjacent_y][adjacent_x]:
                            visited[adjacent] = 1
                            queue.append(adjacent)

            if len(component) > len(largest):
                largest = component

    cleaned = [[False] * width for _ in range(height)]
    for index in largest:
        y, x = divmod(index, width)
        cleaned[y][x] = True
    return cleaned, len(largest)


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
    silhouette = _filled_silhouette(cropped)
    silhouette_area = sum(sum(row) for row in silhouette)
    fill_ratio = silhouette_area / (box_width * box_height)

    hull = _convex_hull(_silhouette_edge_points(silhouette))
    tolerance = max(box_width, box_height) * 0.03
    corners = _simplify_convex_hull(hull, tolerance)

    if len(corners) <= 3:
        return "triangle"

    if len(corners) == 4:
        if _opposite_sides_match(corners):
            return "rectangle"
        # A triangle whose point extends beyond the canvas becomes a trapezoid.
        # Its silhouette still occupies about half of its bounding rectangle.
        if fill_ratio <= 0.68:
            return "triangle"

    # Rasterized circles retain many significant hull points. A heavily clipped
    # triangle can as well, but its convex fill remains much smaller than a disk.
    if fill_ratio <= 0.64:
        return "triangle"
    return "circle"


def _filled_silhouette(mask: list[list[bool]]) -> list[list[bool]]:
    """Fill the outer horizontal span so outlined and solid shapes compare alike."""

    silhouette: list[list[bool]] = []
    for row in mask:
        positions = [index for index, value in enumerate(row) if value]
        filled = [False] * len(row)
        if positions:
            filled[positions[0] : positions[-1] + 1] = [True] * (
                positions[-1] - positions[0] + 1
            )
        silhouette.append(filled)
    return silhouette


def _silhouette_edge_points(mask: list[list[bool]]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for y, row in enumerate(mask):
        positions = [x for x, value in enumerate(row) if value]
        if positions:
            points.append((positions[0], y))
            points.append((positions[-1], y))
    return points


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(
        origin: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _simplify_convex_hull(
    hull: list[tuple[int, int]], tolerance: float
) -> list[tuple[int, int]]:
    vertices = list(hull)
    while len(vertices) > 3:
        distances = [
            _point_line_distance(
                vertices[index],
                vertices[index - 1],
                vertices[(index + 1) % len(vertices)],
            )
            for index in range(len(vertices))
        ]
        smallest = min(distances)
        if smallest > tolerance:
            break
        vertices.pop(distances.index(smallest))
    return vertices


def _point_line_distance(
    point: tuple[int, int], start: tuple[int, int], end: tuple[int, int]
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    numerator = abs(
        delta_y * point[0]
        - delta_x * point[1]
        + end[0] * start[1]
        - end[1] * start[0]
    )
    return numerator / max(1.0, math.hypot(delta_x, delta_y))


def _opposite_sides_match(corners: list[tuple[int, int]]) -> bool:
    lengths = [
        math.dist(corners[index], corners[(index + 1) % 4]) for index in range(4)
    ]

    def ratio(first: float, second: float) -> float:
        return min(first, second) / max(first, second, 1.0)

    return ratio(lengths[0], lengths[2]) >= 0.70 and ratio(
        lengths[1], lengths[3]
    ) >= 0.70


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
