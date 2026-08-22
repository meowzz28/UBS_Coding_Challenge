import asyncio
import base64
import random
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
            assert set(properties) == {"expression"}
            assert "standard precedence" in properties["expression"]["description"]

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
