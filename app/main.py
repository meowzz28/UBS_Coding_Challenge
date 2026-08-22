"""Single FastAPI entry point for all UBS coding challenges."""

from fastapi import FastAPI

from app.challenges import adaptive_api, ghost_chains

app = FastAPI(
    title="UBS Coding Challenge Server",
    description="One deployable FastAPI service containing independent challenge routers.",
    version="2.0.0",
)

app.include_router(adaptive_api.router)
app.include_router(ghost_chains.router)


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
        },
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    return {"status": "ok"}
