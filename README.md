# UBS Coding Challenge Server

One FastAPI service for multiple UBS coding challenges. Each challenge lives in
its own router, while a single `app.main:app` process is deployed to Render.

## Repository structure

```text
app/
├── main.py                         # FastAPI entry point and router registration
└── challenges/
    ├── adaptive_api.py             # Adaptive API Gateway
    └── ghost_chains.py             # Ghost Chains Phase 1
tests/
├── test_adaptive_api.py
└── test_ghost_chains.py
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
git commit -m "Add modular Ghost Chains Phase 1 API"
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

## Add the next challenge

1. Create `app/challenges/<challenge_name>.py` with an `APIRouter` using a unique
   prefix such as `/<challenge-name>`.
2. Import the module in `app/challenges/__init__.py`.
3. Register its router in `app/main.py`.
4. Add an independent test file under `tests/`.
5. Push to `main`; Render deploys the updated single service automatically.

The Render start command remains unchanged:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
