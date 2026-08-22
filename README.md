# UBS Coding Challenge Server

One FastAPI service for multiple UBS coding challenges. Each challenge lives in
its own router, while a single `app.main:app` process is deployed to Render.

## Repository structure

```text
app/
├── main.py                         # FastAPI entry point and router registration
└── challenges/
    ├── adaptive_api.py             # Adaptive API Gateway
    ├── ghost_chains.py             # Ghost Chains Phase 1
    ├── tool_box.py                 # Tool Box Phase 1 MCP server
    └── showdown.py                 # SHOWDOWN Phase 1 betting strategy
tests/
├── test_adaptive_api.py
├── test_ghost_chains.py
├── test_tool_box.py
└── test_showdown.py
render.yaml                         # Existing Render service configuration
```

This structure keeps challenge models, state, algorithms, and routes separate.
Adding another challenge does not require another server or Render deployment.

## Available APIs

| Challenge | Method | Endpoint |
| --- | --- | --- |
| Service | `GET` | `/health` |
| Adaptive API | `POST` | `/solve` |
| Adaptive API | `POST` | `/adaptive-api/solve` |
| Ghost Chains | `GET` | `/ghost-chains/health` |
| Ghost Chains | `POST` | `/ghost-chains/reset` |
| Ghost Chains | `POST` | `/ghost-chains/transactions` |
| Tool Box | MCP | `/mcp` |
| Tool Box | `GET` | `/tool-box/health` |
| SHOWDOWN | `POST` | `/move` |
| SHOWDOWN | `POST` | `/showdown/move` |
| SHOWDOWN | `GET` | `/showdown/health` |

The original `/solve` endpoint remains unchanged so the already-configured
Adaptive API evaluation continues to work. New challenges use a namespaced path
to avoid route collisions.

## Run locally

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

The API runs at <http://127.0.0.1:8000>. Interactive OpenAPI documentation is at
<http://127.0.0.1:8000/docs>.

Run every challenge test:

```bash
pytest
```

## Adaptive API Gateway

`POST /solve` and `POST /adaptive-api/solve` both decode the Base64 payload and
perform the original transformation:

- `adaptInput.user.id` becomes `adaptOutput.id`
- `adaptInput.user.fullName` becomes `adaptOutput.name`
- `adaptInput.action` is converted to lowercase
- priorities map as `LOW = 1`, `MEDIUM = 2`, and `HIGH = 3`

## Ghost Chains - Phase 1

The Ghost Chains implementation maintains an in-memory directed transaction
multigraph. It processes each request array sequentially and scores the current
transaction before adding it to the graph.

The Phase 1 score combines structural signals including:

- isolated transfers and ordinary one-way extensions
- shared-ancestor convergence
- new direct edges that shorten existing paths
- return edges that close directed cycles
- additional return routes into nodes already participating in cycles

Scores are deterministic numbers from `0.0` through `1.0`. Transactions older
than the 24-hour watermark are removed from graph state. An identical duplicate
`txId` returns its original score without mutating state; reusing a `txId` with a
different payload returns HTTP `409`. Optional and unknown fields are accepted so
later phases can extend the transaction model.

### Smoke test

Reset state before a new evaluation:

```bash
curl -X POST http://127.0.0.1:8000/ghost-chains/reset \
  -H 'Content-Type: application/json' \
  -d '{"clearTransactions":true}'
```

Submit an ordered transaction batch:

```bash
curl -X POST http://127.0.0.1:8000/ghost-chains/transactions \
  -H 'Content-Type: application/json' \
  -d '{
    "transactions": [
      {
        "txId": "tx_meridian_001",
        "fromUserId": "meridian_holdings",
        "toUserId": "apex_logistics",
        "amount": 370.0,
        "createdAt": "2026-06-08T12:00:00Z"
      }
    ]
  }'
```

Response shape:

```json
{
  "transactions": [
    {"txId": "tx_meridian_001", "riskScore": 0.02}
  ]
}
```

## Deploy updates to the existing Render service

No new server is needed. Commit and push the new router to the same `main` branch:

```bash
git add .
git commit -m "Add the next challenge API"
git push origin main
```

Render's automatic deployment will update the existing service. Keep the same
base URL and register these Ghost Chains endpoints beneath it:

```text
https://YOUR-EXISTING-SERVICE.onrender.com/ghost-chains/health
https://YOUR-EXISTING-SERVICE.onrender.com/ghost-chains/reset
https://YOUR-EXISTING-SERVICE.onrender.com/ghost-chains/transactions
```

If the coordinator asks for a public base URL instead of individual endpoints,
provide:

```text
https://YOUR-EXISTING-SERVICE.onrender.com
```

The service uses one Uvicorn worker so all Ghost Chains requests share the same
in-memory graph. A process restart restores clean startup state, and the evaluator
can explicitly establish that state through `/ghost-chains/reset`.

## Tool Box - Phase 1

Tool Box is exposed through the official MCP Streamable HTTP transport rather
than an ordinary REST endpoint. Register this URL with the evaluator:

```text
https://YOUR-EXISTING-SERVICE.onrender.com/mcp
```

The MCP server advertises three model-callable tools:

| Tool | Purpose |
| --- | --- |
| `get_name` | Returns the valid assigned name `Nova Box` |
| `calculate` | Evaluates a complete `+`, `-`, `*`, `/` expression with precedence |
| `identify_shape` | Classifies a base64 PNG as `rectangle`, `triangle`, or `circle` |

The shape tool supports filled and outlined shapes, arbitrary rotation, clipped
edges, isolated pixel noise, colored or dark foregrounds, transparency, and PNG
data URIs. The server instructions tell the evaluator's multi-turn agent to
compose tools for combination questions and never guess.

Arithmetic expressions are submitted to `calculate` in one call. The evaluator
supports parentheses and standard multiplication/division precedence, while a
restricted Python AST prevents function calls, names, or unsupported operators.
Every integer literal must remain within the Phase 1 range of -100 to 100.

The MCP implementation is stateless and returns JSON responses, so individual
tool calls do not depend on process affinity. The FastAPI lifespan starts the MCP
session manager used by legacy MCP clients, and the SDK also supports the current
sessionless protocol on the same `/mcp` endpoint.

The server advertises only three concise tools, well below the challenge limit
of 20, and every tool returns a tiny result well below the 1,200-token limit.
All computation is local and comfortably inside the 10-second response limit.

## SHOWDOWN - Phase 1

Register the existing Render base URL with the SHOWDOWN evaluator. It calls:

```text
POST https://YOUR-EXISTING-SERVICE.onrender.com/move
```

The bot is stateless and deterministic, so repeated delivery of the same turn
produces the same move. Its strategy combines exact one-card showdown equity,
pot odds, position, current stack risk, legal raise bounds, and opponent
tendencies reconstructed from the supplied rolling history. It value-bets pairs
and premium numbers, avoids high-risk marginal calls, and bluffs only at a
frequency supported by observed folds.

Every response is selected from `legal_actions`; `amount` is returned only for
`bet` or `raise` and is clamped to the coordinator-provided inclusive range.
`GET /health` warms the whole service, while `GET /showdown/health` is available
for challenge-specific checks.

## Add the next challenge

1. Create `app/challenges/<challenge_name>.py` with either an `APIRouter` for REST
   or an `MCPServer` when the challenge requires MCP.
2. Import the module in `app/challenges/__init__.py`.
3. Register its router in `app/main.py`.
4. Add an independent test file under `tests/`.
5. Push to `main`; Render deploys the updated single service automatically.

The Render start command remains unchanged:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
