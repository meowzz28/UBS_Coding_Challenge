# UBS Coding Challenge Server

One FastAPI service for multiple UBS coding challenges. Each challenge lives in
its own router, while a single `app.main:app` process is deployed to Render.

## Repository structure

```text
app/
├── main.py                         # FastAPI entry point and router registration
└── challenges/
    ├── adaptive_api.py             # Adaptive API Gateway
    ├── ghost_chains.py             # Ghost Chains Phases 1-2
    ├── tool_box.py                 # Tool Box Phase 1-2 MCP server
    └── showdown.py                 # SHOWDOWN Phase 1-2 betting strategy
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

Phase 2 payloads may also include `heartbeats` and an `sloQuery`. The server
filters heartbeats to the requested service with `timestamp >= since`, calculates
availability as the fraction whose status is `OK`, and reports the nearest-rank
95th-percentile latency. If the filtered window is empty, both metrics are zero.
Phase 1 payloads remain supported and omit `sloOutput` from the response.

## Ghost Chains - Phases 1 and 2

The Ghost Chains implementation maintains an in-memory directed transaction
multigraph. It processes each request array sequentially and scores the current
transaction before adding it to the graph.

The Phase 1 score combines structural signals including:

- isolated transfers and ordinary one-way extensions
- shared-ancestor convergence
- new direct edges that shorten existing paths
- return edges that close directed cycles
- additional return routes into nodes already participating in cycles

Phase 2 adds an attributed identity layer for `ipAddress` and `deviceId`. The
two dimensions are evaluated independently and combined with graph context. The
model distinguishes consistent identity along a flow from mid-flow changes,
identity dropped on later connected legs, branch divergence, and bounded shared-
infrastructure hints across disconnected components. Identity evidence expires
with the same exact 24-hour window as its transaction edge.

Scores are deterministic numbers from `0.0` through `1.0`. Transactions older
than the 24-hour watermark are removed from graph and identity state. An
identical duplicate `txId` returns its original score without mutating state,
including after graph expiry; reusing a `txId` with a different payload returns
HTTP `409` before any batch mutation. Optional and unknown fields are accepted so
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

## Tool Box - Phases 1, 2, and 3

Tool Box is exposed through the official MCP Streamable HTTP transport rather
than an ordinary REST endpoint. Register this URL with the evaluator:

```text
https://YOUR-EXISTING-SERVICE.onrender.com/mcp
```

The MCP server advertises ten model-callable tools:

| Tool | Purpose |
| --- | --- |
| `get_name` | Returns the valid assigned name `Nova Box` |
| `calculate` | Evaluates a complete `+`, `-`, `*`, `/` expression with precedence |
| `identify_shape` | Classifies a base64 PNG as `rectangle`, `triangle`, or `circle` |
| `search` | Returns relevant passages as a JSON array within the 900-token ceiling |
| `navigate` | Returns the next adjacent node on the least-cost valid route |
| `next_journey_node` | Compatibility name for the same journey operation |
| `find_open_venues` | Returns every venue open at a specified weekday and hour |
| `find_meeting_window` | Applies friend schedules plus accepted/tentative inbox precedence |
| `find_meeting_point` | Minimizes total Manhattan travel for the complete group |
| `plan_outing` | Jointly solves the window, meeting point, and onward eating venue |

The shape tool supports filled and outlined shapes, arbitrary rotation, clipped
edges, isolated pixel noise, colored or dark foregrounds, transparency, and PNG
data URIs. The server instructions tell the evaluator's multi-turn agent to
compose tools for combination questions and never guess.

Arithmetic expressions are submitted to `calculate` in one call. The evaluator
supports parentheses and standard multiplication/division precedence, while a
restricted Python AST prevents function calls, names, or unsupported operators.
Every integer literal must remain within the Phase 1 range of -100 to 100.

For Phase 2 recall, the five published study documents are fetched concurrently,
split into focused passages, and cached. Ranking combines rare-term relevance,
phrase matching, and domain synonym expansion. The response is measured with the
required `o200k_base` tokenizer and can never exceed 900 content tokens.

For journeys, the server retrieves the random directed graph using the opaque
`map_id` and minimizes edge weights plus the toll charged on every entered node.
Hop-limited routes use constrained shortest-path search. Chosen routes are cached
between calls so every returned hop is adjacent, never revisits a node, and stays
within the allowance. Named school-trip destinations can also be resolved from
the study materials, translating documented `STOP_xx` identifiers to the map's
`SITE_x` convention when necessary.

For Phase 3, venue hours, friend schedules, daily locations, and the invitation
inbox are fetched from the authoritative challenge endpoints and cached after
validation. Meeting times use half-open hourly intervals: accepted invitations
and friends' busy intervals are hard conflicts, declined invitations are ignored,
and a later clean window beats every earlier tentative conflict. Grid problems
enumerate the 10 by 10 city and include the android plus every named friend.
`plan_outing` first selects the required meeting window and a venue open for the
full following hour, then minimizes everyone's inbound Manhattan travel plus the
onward trip to that venue as one joint objective.

The FastAPI lifespan starts the MCP session manager used by legacy MCP clients,
and the SDK also supports the current sessionless protocol on `/mcp`.

The server advertises only ten concise tools, below the challenge limit of 20.
Phase 1 and Phase 3 results stay far below 1,200 tokens, and Phase 2 recall remains
below its stricter 900-token ceiling. Remote challenge data is cached after its
first use.

## SHOWDOWN - Phases 1, 2, and 3

Register the existing Render base URL with the SHOWDOWN evaluator. It calls:

```text
POST https://YOUR-EXISTING-SERVICE.onrender.com/move
```

The Phase 1 strategy combines exact one-card showdown equity, pot odds, position,
current stack risk, legal raise bounds, and opponent tendencies reconstructed
from the supplied rolling history. It value-bets pairs and premium numbers,
avoids high-risk marginal calls, and bluffs only at a frequency supported by
observed folds.

For Phases 2 and 3, the bot treats `table_rule` as an opaque stable identifier.
Every revealed showdown filters an exact bank of deterministic rank hypotheses,
including pair/high, lowball, proximity, cyclic, parity, prime, and sum-target
rules. A rule locks only when one hypothesis remains 100% consistent. When the
real rule is outside that bank, a per-community pairwise tournament matrix uses
direct results plus safe transitive closure. Both rule evidence and locked models
remain cached by codename across legs and retries.

Phase 3 equity counts only non-folded, non-busted opponents and computes exact
multiway pot share, including split-pot ties. Persistent per-name profiles track
VPIP, pre-reveal raises, aggression factor, and folds to raises for Dana, Miles,
Theo, Rhea, and Bram. The 60-hand policy changes from early extraction against
aggressive players, to midgame shorthanded value and positional steals, to a
strict endgame leaderboard policy that protects a qualifying 12-chip lead or
attacks the leader with top-quartile holdings when trailing. Premium early and
urgent endgame lines may legally use `max_raise_to`; steals remain near half-pot.
Phase 2 retains its `+25` forced-bet lock and conservative threshold sizing.

Every response is selected from `legal_actions`; `amount` is returned only for
`bet` or `raise` and is clamped to the coordinator-provided inclusive range.
`GET /health` warms the whole service, while `GET /showdown/health` is available
for challenge-specific checks. Render runs one worker so rule and opponent memory
are shared by every move handled by the deployed process.

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
