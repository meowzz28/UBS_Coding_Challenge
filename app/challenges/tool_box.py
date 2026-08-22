"""MCP tools for Tool Box Phases 1 and 2."""

from __future__ import annotations

import ast
import base64
import binascii
import heapq
import json
import math
import re
import statistics
import threading
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from io import BytesIO
from typing import Annotated, Any, Literal, NamedTuple, cast

import httpx
import tiktoken
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
MAX_RECALL_TOKENS = 900
CHALLENGE_ORIGIN = "https://tool-box-2591eaa24fa3.herokuapp.com"
STUDY_MATERIAL_URLS = tuple(
    f"{CHALLENGE_ORIGIN}/study-materials/{document_id}"
    for document_id in range(1, 6)
)
GRAPH_URL = f"{CHALLENGE_ORIGIN}/graph"
HTTP_TIMEOUT_SECONDS = 5.0
_TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


class StudyChunk(NamedTuple):
    document_id: int
    heading: str
    passage: str


_study_chunks: tuple[StudyChunk, ...] | None = None
_study_lock = threading.Lock()
_graph_cache: dict[str, dict[str, Any]] = {}
_route_cache: dict[tuple[str, str], tuple[str, ...]] = {}
_route_lock = threading.Lock()

server = MCPServer(
    name="UBS Tool Box",
    title="Tool Box Phases 1 and 2",
    description=(
        "Tools for identity, arithmetic, shape recognition, study recall, and "
        "least-cost directed-graph journeys."
    ),
    instructions=(
        "Use get_name when asked for your name. For every arithmetic question, "
        "call calculate exactly once with the entire expression; it applies "
        "standard precedence and parentheses, so never split an expression into "
        "sequential calls or evaluate it left-to-right. Use identify_shape for "
        "base64-encoded PNG images. For an exam or factual recall question, call "
        "search with the complete question and answer only from the "
        "returned source passages. For every journey hop, call next_journey_node "
        "with the map_id, current node, actual destination, and the supplied hop "
        "allowance when present; return its node exactly. For a school trip whose "
        "destination is a named place rather than a node, first use search to find "
        "its STOP number, then navigate to that STOP. "
        "Combine tools when a request contains more than one kind of task. Return "
        "tool results directly and do not guess."
    ),
    version="2.1.0",
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
    pixels = cast(Any, image.load())
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


@server.tool()
def search(
    query: Annotated[
        str,
        Field(
            min_length=3,
            max_length=1000,
            description=(
                "The complete factual question exactly as asked. Returns source "
                "passages containing the answer; use their facts in the final answer."
            ),
        ),
    ],
) -> list[str]:
    """Search all assigned study materials for passages that answer the query."""

    return recall_study_material(query)


def recall_study_material(
    question: Annotated[
        str,
        Field(
            min_length=3,
            max_length=1000,
            description=(
                "The complete exam question or school-trip place-name question. "
                "Include names, dates, and subject terms exactly as asked."
            ),
        ),
    ],
) -> list[str]:
    """Return the most relevant source passages within the 900-token recall limit."""

    ranked = _rank_study_chunks(question)
    passages: list[str] = []
    used_tokens = 0

    for _, chunk in ranked:
        passage = chunk.passage.strip()
        token_count = len(_TOKEN_ENCODING.encode(passage))
        remaining = MAX_RECALL_TOKENS - used_tokens
        if token_count > remaining:
            if passages or remaining < 40:
                continue
            passage = _TOKEN_ENCODING.decode(
                _TOKEN_ENCODING.encode(passage)[:remaining]
            ).strip()
            token_count = len(_TOKEN_ENCODING.encode(passage))
        if passage:
            passages.append(passage)
            used_tokens += token_count
        if len(passages) >= 4 or used_tokens >= MAX_RECALL_TOKENS:
            break

    if not passages:
        raise ValueError("no relevant study passage could be retrieved")
    return passages


@server.tool()
def next_journey_node(
    map_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=300,
            description="The opaque map_id copied exactly from the journey question.",
        ),
    ],
    current_node: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200,
            description="The node where the android is currently standing.",
        ),
    ],
    destination: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description=(
                "The destination node. A school-trip place name is also accepted "
                "and will be resolved from the study materials to its STOP number."
            ),
        ),
    ],
    hops_remaining: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Edges still allowed, including the hop being requested. Omit "
                "only when the question gives no hop allowance."
            ),
        ),
    ] = None,
) -> str:
    """Return the next adjacent node on the least-cost valid route."""

    clean_map_id = map_id.strip()
    graph = _get_graph(clean_map_id)
    adjacency, tolls = _validate_graph(graph)
    current = _match_node(current_node, set(tolls), "current_node")
    resolved_destination = _resolve_destination(destination, set(tolls))

    if current == resolved_destination:
        return resolved_destination

    cache_key = (clean_map_id, resolved_destination)
    with _route_lock:
        cached_route = _route_cache.get(cache_key)
        if cached_route and current in cached_route:
            position = cached_route.index(current)
            remaining_route_hops = len(cached_route) - position - 1
            if remaining_route_hops and (
                hops_remaining is None or remaining_route_hops <= hops_remaining
            ):
                return cached_route[position + 1]

    route = _least_cost_route(
        adjacency,
        tolls,
        current,
        resolved_destination,
        hops_remaining,
    )
    with _route_lock:
        _route_cache[cache_key] = tuple(route)
    return route[1]


def _fetch_text(url: str) -> str:
    try:
        response = httpx.get(
            url,
            headers={
                "Accept": "text/plain, text/markdown, application/json",
                "User-Agent": "UBS-Tool-Box/2.0",
            },
            timeout=HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as exc:
        raise ValueError("challenge data source is temporarily unavailable") from exc


def _get_study_chunks() -> tuple[StudyChunk, ...]:
    global _study_chunks
    if _study_chunks is not None:
        return _study_chunks

    with _study_lock:
        if _study_chunks is not None:
            return _study_chunks
        with ThreadPoolExecutor(max_workers=len(STUDY_MATERIAL_URLS)) as executor:
            documents = tuple(executor.map(_fetch_text, STUDY_MATERIAL_URLS))
        chunks = [
            chunk
            for document_id, document in enumerate(documents, start=1)
            for chunk in _chunk_markdown(document_id, document)
        ]
        if not chunks:
            raise ValueError("study materials did not contain any readable passages")
        _study_chunks = tuple(chunks)
        return _study_chunks


def _chunk_markdown(document_id: int, document: str) -> list[StudyChunk]:
    heading = f"Study material {document_id}"
    paragraph_lines: list[str] = []
    chunks: list[StudyChunk] = []

    def finish_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines.clear()
        if not paragraph:
            return
        passage = f"{heading}\n{paragraph}"
        encoded = _TOKEN_ENCODING.encode(passage)
        for offset in range(0, len(encoded), 300):
            part = _TOKEN_ENCODING.decode(encoded[offset : offset + 300]).strip()
            if part:
                chunks.append(StudyChunk(document_id, heading, part))

    for raw_line in document.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            finish_paragraph()
            heading_text = line.lstrip("#").strip()
            if heading_text:
                heading = f"Study material {document_id} — {heading_text}"
        elif not line:
            finish_paragraph()
        else:
            paragraph_lines.append(line)
    finish_paragraph()
    return chunks


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "bring",
    "brought",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "last",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

_SYNONYM_GROUPS = (
    {"align", "alignment", "calibrate", "calibration", "recalibrate"},
    {"array", "grid", "hydrophone", "sensor"},
    {"boss", "chair", "director", "head", "lead", "leader", "investigator"},
    {
        "crew",
        "employee",
        "engineer",
        "member",
        "participant",
        "people",
        "personnel",
        "population",
        "resident",
        "scientist",
        "staff",
        "technician",
        "worker",
    },
    {"cap", "ceiling", "limit", "maximum", "threshold"},
    {"accident", "event", "failure", "fault", "incident"},
    {"schedule", "cadence", "cycle", "frequency", "interval"},
    {"cost", "fare", "fee", "price", "toll"},
    {"code", "callsign", "designation", "identifier", "tag"},
    {"place", "destination", "facility", "location", "premises", "stop"},
    {"vehicle", "craft", "submersible", "train", "trainset"},
    {"fix", "patch", "repair", "restore", "service"},
    {"medicine", "compound", "dose", "drug", "injection"},
    {"experiment", "research", "study", "trial"},
)

_DOCUMENT_HINTS = {
    1: {
        "aboard",
        "abyss",
        "crew",
        "deep",
        "dive",
        "facility",
        "habitat",
        "ocean",
        "outpost",
        "station",
        "submersible",
        "trench",
        "undersea",
        "underwater",
    },
    2: {
        "bus",
        "commuter",
        "driver",
        "fare",
        "passenger",
        "rail",
        "rider",
        "route",
        "train",
        "transit",
        "transport",
    },
    3: {
        "clinical",
        "compound",
        "dose",
        "drug",
        "injection",
        "medical",
        "medicine",
        "patient",
        "participant",
        "study",
        "trial",
        "velmara",
    },
    4: {
        "build",
        "console",
        "engine",
        "game",
        "graphics",
        "physics",
        "renderer",
        "software",
        "texture",
    },
    5: {
        "agriculture",
        "cooperative",
        "crop",
        "farm",
        "farmer",
        "grain",
        "grower",
        "harvest",
        "produce",
        "storage",
    },
}

_SECTION_HINTS = {
    "staffing roster": {
        "aboard",
        "count",
        "crew",
        "how many",
        "live",
        "people",
        "personnel",
        "population",
        "resident",
        "simultaneously",
        "staff",
    },
    "membership roster": {
        "count",
        "household",
        "how many",
        "member",
        "membership",
        "people",
    },
    "leadership and team structure": {
        "count",
        "engineer",
        "how many",
        "personnel",
        "simultaneously",
        "staff",
        "team",
    },
    "cohort structure": {
        "cohort",
        "enrolled",
        "how many",
        "participant",
        "recruitment",
    },
    "leadership": {
        "boss",
        "charge",
        "director",
        "head",
        "lead",
        "leader",
        "manage",
        "responsible",
        "run",
    },
    "incident": {"accident", "date", "failure", "fault", "happened", "incident"},
    "calibration": {"alignment", "array", "calibrate", "grid", "sensor"},
    "schedule": {"cadence", "cycle", "frequency", "how often", "interval", "schedule"},
}


def _stem(token: str) -> str:
    if token.startswith(("align", "calibr", "recalibr")):
        return "calibr"
    if token == "runs":
        return "run"
    for suffix in ("ation", "ments", "ment", "ingly", "edly", "ing", "ies", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def _terms(text: str) -> list[str]:
    return [
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOPWORDS
    ]


def _expanded_query_terms(question: str) -> dict[str, float]:
    original = set(_terms(question))
    weighted = {term: 1.0 for term in original}
    for group in _SYNONYM_GROUPS:
        stemmed_group = {_stem(term) for term in group}
        if original & stemmed_group:
            for term in stemmed_group:
                weighted.setdefault(term, 0.45)
    return weighted


def _rank_study_chunks(question: str) -> list[tuple[float, StudyChunk]]:
    chunks = _get_study_chunks()
    query_terms = _expanded_query_terms(question)
    term_sets = [set(_terms(chunk.passage)) for chunk in chunks]
    document_frequency = {
        term: sum(term in terms for terms in term_sets) for term in query_terms
    }
    question_words = _terms(question)
    question_term_set = set(question_words)
    bigrams = {
        f"{question_words[index]} {question_words[index + 1]}"
        for index in range(len(question_words) - 1)
    }

    ranked: list[tuple[float, StudyChunk]] = []
    for chunk, chunk_terms in zip(chunks, term_sets, strict=True):
        normalized = " ".join(_terms(chunk.passage))
        normalized_question = " ".join(question_words)
        score = 0.0
        for term, query_weight in query_terms.items():
            if term in chunk_terms:
                rarity = math.log((len(chunks) + 1) / (document_frequency[term] + 1)) + 1
                score += query_weight * rarity
        document_hints = {_stem(term) for term in _DOCUMENT_HINTS[chunk.document_id]}
        score += 2.4 * len(question_term_set & document_hints)
        heading = chunk.heading.casefold()
        for heading_fragment, raw_hints in _SECTION_HINTS.items():
            if heading_fragment in heading:
                hints = {
                    stemmed
                    for hint in raw_hints
                    for stemmed in _terms(hint)
                }
                score += 3.2 * len(question_term_set & hints)
        if any(
            intent in normalized_question
            for intent in ("charge", "lead", "manage", "responsible", "run")
        ) and any(
            marker in chunk.passage.casefold()
            for marker in (
                "director-general",
                "elected chair",
                "has served as",
                "holds primary responsibility",
                "lead architect",
                "presided over",
                "station director",
            )
        ):
            score += 12.0
        score += 1.8 * sum(bigram in normalized for bigram in bigrams)
        stop_matches = set(re.findall(r"stop[_ -]?\d+", question.casefold()))
        if stop_matches and any(match.replace(" ", "_").replace("-", "_") in chunk.passage.casefold() for match in stop_matches):
            score += 20.0
        ranked.append((score, chunk))

    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1].document_id,
        )
    )
    return ranked


def _get_graph(map_id: str) -> dict[str, Any]:
    with _route_lock:
        cached = _graph_cache.get(map_id)
    if cached is not None:
        return cached

    url = f"{GRAPH_URL}?{urllib.parse.urlencode({'map_id': map_id})}"
    raw = _fetch_text(url)
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("map source returned invalid JSON") from exc
    if not isinstance(graph, dict):
        raise ValueError("map source returned an invalid graph")
    with _route_lock:
        _graph_cache[map_id] = graph
    return graph


def _validate_graph(
    graph: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    raw_adjacency = graph.get("adjacency")
    raw_tolls = graph.get("tolls")
    if not isinstance(raw_adjacency, dict) or not isinstance(raw_tolls, dict):
        raise ValueError("map must contain adjacency and tolls objects")

    tolls: dict[str, float] = {}
    for node, value in raw_tolls.items():
        if not isinstance(node, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("map tolls must be finite non-negative numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("map tolls must be finite non-negative numbers")
        tolls[node] = number

    adjacency: dict[str, dict[str, float]] = {}
    for node, neighbors in raw_adjacency.items():
        if not isinstance(node, str) or not isinstance(neighbors, dict):
            raise ValueError("map adjacency must map nodes to neighbor objects")
        adjacency[node] = {}
        for neighbor, value in neighbors.items():
            if (
                not isinstance(neighbor, str)
                or neighbor not in tolls
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError("map edges must target known nodes with numeric weights")
            weight = float(value)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("map edge weights must be finite and non-negative")
            adjacency[node][neighbor] = weight
    if not tolls or set(adjacency) != set(tolls):
        raise ValueError("map must list every node in adjacency and tolls")
    return adjacency, tolls


def _match_node(value: str, nodes: set[str], field_name: str) -> str:
    candidate = value.strip()
    if candidate in nodes:
        return candidate
    matches = [node for node in nodes if node.casefold() == candidate.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"{field_name} is not a node in this map")


def _resolve_destination(destination: str, nodes: set[str]) -> str:
    candidate = destination.strip()
    try:
        return _match_node(candidate, nodes, "destination")
    except ValueError:
        pass

    explicit_stops = re.findall(r"STOP[_ -]?\d+", candidate, flags=re.IGNORECASE)
    for explicit_stop in explicit_stops:
        for node_candidate in _stop_node_candidates(explicit_stop):
            try:
                return _match_node(node_candidate, nodes, "destination")
            except ValueError:
                continue

    for _, chunk in _rank_study_chunks(candidate)[:12]:
        for stop in re.findall(r"STOP_\d+", chunk.passage, flags=re.IGNORECASE):
            for node_candidate in _stop_node_candidates(stop):
                try:
                    return _match_node(node_candidate, nodes, "destination")
                except ValueError:
                    continue
    raise ValueError("destination could not be resolved to a node in this map")


def _stop_node_candidates(stop: str) -> tuple[str, ...]:
    """Accept both document STOP_05 and journey-map SITE_5 naming conventions."""

    match = re.search(r"\d+", stop)
    if match is None:
        raise ValueError("stop identifier must contain a number")
    number = int(match.group())
    return (
        f"STOP_{number:02d}",
        f"STOP_{number}",
        f"SITE_{number:02d}",
        f"SITE_{number}",
    )


def _least_cost_route(
    adjacency: dict[str, dict[str, float]],
    tolls: dict[str, float],
    start: str,
    destination: str,
    hops_remaining: int | None,
) -> list[str]:
    maximum_hops = min(
        hops_remaining if hops_remaining is not None else len(tolls) - 1,
        len(tolls) - 1,
    )
    queue: list[tuple[float, int, tuple[str, ...], str]] = [
        (0.0, 0, (start,), start)
    ]
    best: dict[tuple[str, int], float] = {(start, 0): 0.0}

    while queue:
        cost, hops, path, node = heapq.heappop(queue)
        if cost > best.get((node, hops), math.inf):
            continue
        if node == destination:
            return list(path)
        if hops >= maximum_hops:
            continue

        for neighbor, edge_weight in sorted(adjacency[node].items()):
            if neighbor in path:
                continue
            next_hops = hops + 1
            next_cost = cost + edge_weight + tolls[neighbor]
            state = (neighbor, next_hops)
            if next_cost < best.get(state, math.inf):
                best[state] = next_cost
                heapq.heappush(
                    queue,
                    (next_cost, next_hops, path + (neighbor,), neighbor),
                )

    if hops_remaining is None:
        raise ValueError("destination is unreachable from the current node")
    raise ValueError("destination cannot be reached within the remaining hops")


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
