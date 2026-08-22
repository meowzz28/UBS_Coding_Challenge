"""Single FastAPI entry point for all UBS coding challenges."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.challenges import adaptive_api, ghost_chains, tool_box


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    async with tool_box.server.session_manager.run():
        yield

app = FastAPI(
    title="UBS Coding Challenge Server",
    description="One deployable FastAPI service containing independent challenge routers.",
    version="3.0.0",
    lifespan=lifespan,
)

app.include_router(adaptive_api.router)
app.include_router(ghost_chains.router)
app.include_router(tool_box.router)


@app.get("/", tags=["service"])
def service_info() -> dict[str, object]:
    return {
        "service": "UBS Coding Challenge Server",
        "status": "ready",
        "challenges": {
            "adaptiveApi": ["/solve", "/adaptive-api/solve"],
            "ghostChains": [
                "/ghost-chains/health",
                "/ghost-chains/reset",
                "/ghost-chains/transactions",
            ],
            "toolBox": ["/mcp", "/tool-box/health"],
        },
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    return {"status": "ok"}


# Keep this catch-all mount last so the REST challenge routes above retain
# precedence while the MCP SDK owns the exact /mcp protocol endpoint.
app.mount("/", tool_box.http_app)
