import asyncio
import base64
import json
import random
from io import BytesIO

import pytest
import tiktoken
from fastapi.testclient import TestClient
from mcp import Client
from PIL import Image, ImageDraw

from app.challenges import tool_box
from app.challenges.tool_box import (
    ASSISTANT_NAME,
    StudyChunk,
    _least_cost_route,
    calculate,
    identify_shape,
    navigate,
    next_journey_node,
    recall_study_material,
    search,
    server,
)
from app.main import app


def png_base64(
    shape: str,
    *,
    outline: bool = False,
    rotation: float = 0,
    noisy: bool = False,
    clipped: bool = False,
) -> str:
    image = Image.new("RGB", (160, 160), "white")
    shape_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(shape_layer)
    fill = None if outline else "#2166d1"
    width = 5 if outline else 1
    if shape == "rectangle":
        draw.rectangle((35, 45, 125, 115), fill=fill, outline="#2166d1", width=width)
    elif shape == "circle":
        draw.ellipse((35, 35, 125, 125), fill=fill, outline="#2166d1", width=width)
    elif shape == "triangle":
        points = ((40, 25), (135, 80), (40, 135))
        if clipped:
            points = ((55, 20), (178, 80), (55, 158))
        draw.polygon(points, fill=fill, outline="#2166d1", width=width)
    else:
        raise AssertionError(f"unknown test shape: {shape}")

    if rotation:
        shape_layer = shape_layer.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            center=(80, 80),
        )
    image.paste(shape_layer, mask=shape_layer.getchannel("A"))

    if noisy:
        noise = random.Random(20260822)
        pixels = image.load()
        for _ in range(220):
            x = noise.randrange(image.width)
            y = noise.randrange(image.height)
            if pixels[x, y] == (255, 255, 255):
                shade = noise.randrange(0, 190)
                pixels[x, y] = (shade, shade, shade)

    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_name_meets_phase_1_constraints() -> None:
    assert 3 <= len(ASSISTANT_NAME) <= 30
    assert all(character.isalnum() or character in " _-'" for character in ASSISTANT_NAME)


def test_all_arithmetic_operators() -> None:
    assert calculate("2 + 2") == 4
    assert calculate("7 - 10") == -3
    assert calculate("-4 * 5") == -20
    assert calculate("7 / 2") == 3.5


def test_complete_expression_uses_standard_precedence() -> None:
    assert calculate("2 + 3 * 5") == 17
    assert calculate("(2 + 3) * 5") == 25
    assert calculate("100 / -4 + 3 * 2") == -19
    assert calculate("What is 2 + 3 * 5?") == 17


def test_division_by_zero_has_a_clear_error() -> None:
    try:
        calculate("2 / 0")
    except ValueError as exc:
        assert str(exc) == "division by zero is undefined"
    else:
        raise AssertionError("division by zero should fail")


def test_calculator_rejects_non_phase_1_syntax() -> None:
    for expression in ("2 ** 3", "101 + 1", "1.5 + 2", "sum([1, 2])"):
        try:
            calculate(expression)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported expression was accepted: {expression}")


def test_filled_and_outlined_shapes_are_identified() -> None:
    for shape in ("rectangle", "triangle", "circle"):
        assert identify_shape(png_base64(shape)) == shape
        assert identify_shape(png_base64(shape, outline=True)) == shape


def test_shapes_are_rotation_independent_and_ignore_pixel_noise() -> None:
    cases = (
        ("rectangle", 37),
        ("rectangle", 90),
        ("triangle", 25),
        ("triangle", 90),
        ("circle", 0),
    )
    for shape, rotation in cases:
        assert identify_shape(png_base64(shape, rotation=rotation, noisy=True)) == shape


def test_clipped_noisy_triangle_from_failed_evaluation_is_identified() -> None:
    encoded = png_base64("triangle", clipped=True, noisy=True)
    assert identify_shape(encoded) == "triangle"


def test_data_uri_is_accepted() -> None:
    encoded = f"data:image/png;base64,{png_base64('circle')}"
    assert identify_shape(encoded) == "circle"


def test_recall_returns_relevant_passages_within_exact_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filler = "Additional unrelated operating detail. " * 70
    chunks = (
        StudyChunk(1, "Calibration", "Calibration\nThe Kesterline hydrophone array was recalibrated on 14 March."),
        StudyChunk(2, "Fares", "Fares\nThe daily transit fare cap is four pounds ninety."),
        *(StudyChunk(3, "Background", f"Background {index}\n{filler}") for index in range(8)),
    )
    monkeypatch.setattr(tool_box, "_study_chunks", chunks)

    passages = recall_study_material(
        "When was the sensor grid last brought back into alignment?"
    )

    assert isinstance(passages, list)
    assert all(isinstance(passage, str) for passage in passages)
    assert "14 March" in passages[0]
    encoding = tiktoken.get_encoding("o200k_base")
    assert sum(len(encoding.encode(passage)) for passage in passages) <= 900


def test_search_contract_and_indirect_facility_staffing_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_box,
        "_study_chunks",
        (
            StudyChunk(
                1,
                "Study material 1 — Staffing Roster",
                "Study material 1 — Staffing Roster\nThe station maintains a resident population of forty-one scientists and technicians, spread across three rotating shifts so the habitat is never left unstaffed.",
            ),
            StudyChunk(
                4,
                "Study material 4 — Leadership and Team Structure",
                "Study material 4 — Leadership and Team Structure\nThe core engine group maintains thirty-two engineers working simultaneously across four teams.",
            ),
        ),
    )

    passages = json.loads(
        search("Roughly how many personnel live aboard the facility simultaneously?")
    )

    assert "forty-one" in passages[0]


def test_search_returns_one_json_array_of_strings_for_the_grader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_box,
        "_study_chunks",
        (
            StudyChunk(
                1,
                "Equipment Maintenance Logs",
                "The station's primary submersible is named the Halcyon Drift.",
            ),
        ),
    )

    raw_result = search(
        "What is the name given to the outpost's main exploration craft?"
    )
    passages = json.loads(raw_result)

    assert isinstance(passages, list)
    assert all(isinstance(passage, str) for passage in passages)
    assert "Halcyon Drift" in passages[0]


def test_least_cost_route_includes_entry_tolls_and_standard_hop_limit() -> None:
    adjacency = {
        "A": {"B": 4.0, "C": 2.0, "D": 20.0},
        "B": {"D": 3.0},
        "C": {"D": 2.0},
        "D": {},
    }
    tolls = {"A": 5.0, "B": 1.0, "C": 9.0, "D": 2.0}

    assert _least_cost_route(adjacency, tolls, "A", "D", None) == ["A", "B", "D"]
    assert _least_cost_route(adjacency, tolls, "A", "D", 1) == ["A", "D"]


def test_hop_limit_fails_instead_of_returning_an_invalid_route() -> None:
    adjacency = {"A": {"B": 1.0}, "B": {"D": 1.0}, "D": {}}
    tolls = {"A": 0.0, "B": 0.0, "D": 0.0}

    with pytest.raises(ValueError, match="remaining hops"):
        _least_cost_route(adjacency, tolls, "A", "D", 1)


def test_next_journey_node_is_adjacent_and_reuses_the_chosen_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {
        "adjacency": {
            "A": {"B": 4.0, "C": 2.0},
            "B": {"D": 3.0},
            "C": {"D": 2.0},
            "D": {},
        },
        "tolls": {"A": 5.0, "B": 1.0, "C": 9.0, "D": 2.0},
    }
    monkeypatch.setattr(tool_box, "_get_graph", lambda _: graph)
    tool_box._route_cache.clear()

    assert next_journey_node("map-1", "A", "D") == "B"
    assert next_journey_node("map-1", "B", "D") == "D"
    assert navigate("map-1", "A", "D") == "B"


def test_school_trip_place_name_resolves_to_documented_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {
        "adjacency": {"SITE_1": {"SITE_5": 2.0}, "SITE_5": {}},
        "tolls": {"SITE_1": 0.0, "SITE_5": 1.0},
    }
    monkeypatch.setattr(tool_box, "_get_graph", lambda _: graph)
    monkeypatch.setattr(
        tool_box,
        "_study_chunks",
        (
            StudyChunk(
                2,
                "Line Timetables",
                "Line Timetables\nThe Verity Observatory is served by STOP_05 on the Russet Line.",
            ),
        ),
    )
    tool_box._route_cache.clear()

    assert next_journey_node("trip-map", "SITE_1", "Verity Observatory") == "SITE_5"


def test_mcp_server_lists_and_calls_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tool_box,
        "_study_chunks",
        (
            StudyChunk(
                2,
                "Line Timetables",
                "Line Timetables\nThe Verity Observatory is served by STOP_05.",
            ),
        ),
    )
    monkeypatch.setitem(
        tool_box._graph_cache,
        "trip-map",
        {
            "adjacency": {"START": {"STOP_05": 1.0}, "STOP_05": {}},
            "tolls": {"START": 0.0, "STOP_05": 0.0},
        },
    )
    tool_box._route_cache.clear()

    async def scenario() -> None:
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "get_name",
                "calculate",
                "identify_shape",
                "search",
                "navigate",
                "next_journey_node",
            }
            calculate_tool = next(
                tool for tool in listed.tools if tool.name == "calculate"
            )
            properties = calculate_tool.input_schema["properties"]
            assert set(properties) == {"expression"}
            assert "standard precedence" in properties["expression"]["description"]
            search_tool = next(tool for tool in listed.tools if tool.name == "search")
            assert set(search_tool.input_schema["properties"]) == {"query"}

            name_result = await client.call_tool("get_name", {})
            assert name_result.structured_content == {"result": ASSISTANT_NAME}

            math_result = await client.call_tool(
                "calculate",
                {"expression": "2 + 3 * 5"},
            )
            assert math_result.structured_content == {"result": 17}

            shape_result = await client.call_tool(
                "identify_shape",
                {"image_base64": png_base64("triangle")},
            )
            assert shape_result.structured_content == {"result": "triangle"}

            recall_result = await client.call_tool(
                "search",
                {"query": "Which stop serves the Verity Observatory?"},
            )
            assert len(recall_result.content) == 1
            raw_recall = recall_result.content[0].text
            passages = json.loads(raw_recall)
            assert all(isinstance(passage, str) for passage in passages)
            assert "STOP_05" in passages[0]

            navigate_result = await client.call_tool(
                "navigate",
                {
                    "map_id": "trip-map",
                    "current_node": "START",
                    "destination": "STOP_05",
                },
            )
            assert navigate_result.structured_content == {"result": "STOP_05"}

            journey_result = await client.call_tool(
                "next_journey_node",
                {
                    "map_id": "trip-map",
                    "current_node": "START",
                    "destination": "STOP_05",
                },
            )
            assert journey_result.structured_content == {"result": "STOP_05"}

    asyncio.run(scenario())


def test_streamable_http_endpoint_negotiates_at_exact_mcp_path() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    with TestClient(app) as client:
        response = client.post("/mcp", json=request, headers=headers)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["serverInfo"]["name"] == "UBS Tool Box"
    assert result["capabilities"]["tools"] == {"listChanged": False}
