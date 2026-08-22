import asyncio
import base64
from io import BytesIO

from fastapi.testclient import TestClient
from mcp import Client
from PIL import Image, ImageDraw

from app.challenges.tool_box import (
    ASSISTANT_NAME,
    calculate,
    identify_shape,
    server,
)
from app.main import app


def png_base64(shape: str, *, outline: bool = False) -> str:
    image = Image.new("RGB", (120, 100), "white")
    draw = ImageDraw.Draw(image)
    fill = None if outline else "#2166d1"
    width = 5 if outline else 1
    if shape == "rectangle":
        draw.rectangle((20, 20, 100, 80), fill=fill, outline="#2166d1", width=width)
    elif shape == "circle":
        draw.ellipse((25, 10, 95, 80), fill=fill, outline="#2166d1", width=width)
    elif shape == "triangle":
        draw.polygon(
            ((60, 10), (105, 85), (15, 85)),
            fill=fill,
            outline="#2166d1",
            width=width,
        )
    else:
        raise AssertionError(f"unknown test shape: {shape}")

    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_name_meets_phase_1_constraints() -> None:
    assert 3 <= len(ASSISTANT_NAME) <= 30
    assert all(character.isalnum() or character in " _-'" for character in ASSISTANT_NAME)


def test_all_arithmetic_operators() -> None:
    assert calculate(2, "+", 2) == 4
    assert calculate(7, "-", 10) == -3
    assert calculate(-4, "*", 5) == -20
    assert calculate(7, "/", 2) == 3.5


def test_division_by_zero_has_a_clear_error() -> None:
    try:
        calculate(2, "/", 0)
    except ValueError as exc:
        assert str(exc) == "division by zero is undefined"
    else:
        raise AssertionError("division by zero should fail")


def test_filled_and_outlined_shapes_are_identified() -> None:
    for shape in ("rectangle", "triangle", "circle"):
        assert identify_shape(png_base64(shape)) == shape
        assert identify_shape(png_base64(shape, outline=True)) == shape


def test_data_uri_is_accepted() -> None:
    encoded = f"data:image/png;base64,{png_base64('circle')}"
    assert identify_shape(encoded) == "circle"


def test_mcp_server_lists_and_calls_tools() -> None:
    async def scenario() -> None:
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "get_name",
                "calculate",
                "identify_shape",
            }
            calculate_tool = next(
                tool for tool in listed.tools if tool.name == "calculate"
            )
            properties = calculate_tool.input_schema["properties"]
            assert properties["left"]["minimum"] == -100
            assert properties["right"]["maximum"] == 100
            assert properties["operator"]["enum"] == ["+", "-", "*", "/"]

            name_result = await client.call_tool("get_name", {})
            assert name_result.structured_content == {"result": ASSISTANT_NAME}

            math_result = await client.call_tool(
                "calculate",
                {"left": 12, "operator": "*", "right": -3},
            )
            assert math_result.structured_content == {"result": -36}

            shape_result = await client.call_tool(
                "identify_shape",
                {"image_base64": png_base64("triangle")},
            )
            assert shape_result.structured_content == {"result": "triangle"}

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
