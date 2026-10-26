# Tahoe House Operator — MVP

VS Code-friendly starter for the compliance-and-calendar MVP described in the proposal.

## MVP boundary

- LangGraph routes a request through deterministic compliance and calendar slices, a code-based critic, and a reporter.
- One House MCP server is the only tool boundary. It owns permit rules, calendar events, corpus search, and the local database.
- A small synthetic fixture set is included. It is not real property, guest, or financial data.
- The corpus supports local Chroma vector search when the model gateway has an embeddings model configured. Keyword fallback is available for offline development.
- Free-text chat uses the configured gateway chat model to extract a typed plan. The plan can call only allow-listed House MCP tools; code formats the answer from tool results and cited corpus passages.
- Chat history is short-term, process-local context only. It is not added to property rules or long-term memory.
- Free-text messages and recent chat turns are sent to the configured gateway for parsing; do not enter real guest or financial data in this synthetic MVP.
- Calendar changes are proposals unless explicitly approved. The MVP does not send messages, modify listings, charge cards, or remit taxes.
- The gateway is configurable; no model provider or credential is hard-coded. The deterministic workflow does not require a chat model.

## Requirements

- Python 3.11 or 3.12 (3.12 is pinned in `.python-version`)
- VS Code with the Python and Pylance extensions
- `uv` for environment and dependency management
- Gateway credentials and an OpenAI-compatible base URL are required for free-text chat. They remain optional for the deterministic `/run` endpoint and keyword-only RAG. Keep credentials only in `.env`.

On macOS, install Python/uv if needed:

```bash
brew install uv
uv python install 3.12
```

## Setup

Open `tahoe-house-operator/` in VS Code, then run:

```bash
uv sync --extra dev
cp .env.example .env
```

Edit `.env` with the gateway URL, API key, chat model (`MODEL_NAME`), and—optionally—embedding model available to you. The `/chat` endpoint requires the chat model. If the embedding model is blank, RAG uses keyword fallback. Never commit `.env`.

To explicitly build or refresh the local vector index after adding corpus Markdown files:

```bash
uv run python scripts/ingest_corpus.py
```

## Run in VS Code

Use **Run and Debug → Tahoe MVP (API + House MCP)**. This starts:

- Operator API and simple trace client: <http://127.0.0.1:8000>
- House MCP server: <http://127.0.0.1:8001/mcp>

The chat panel accepts free text about occupancy, availability, stay checks, trash, parking, and stored rules. It reports pricing, turnover, guest messaging, or finance as not implemented rather than making up results.

Alternatively start two integrated terminals:

```bash
uv run python -m tahoe_operator.mcp_server
uv run uvicorn tahoe_operator.api:app --reload --host 127.0.0.1 --port 8000
```

Try the synthetic demo request from the page or call the API:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"prompt":"Check a Saturday owner stay and the storm/trash rules","start_date":"2026-11-14","end_date":"2026-11-15","trash_date":"2026-11-13","guest_count":2,"renter_age":35,"vehicles":1,"proposed_action":"owner_stay"}'
```

The fixture intentionally has an owner calendar block on 2026-11-14 and a storm/trash check on 2026-11-13 to exercise critic failures. Use `operator_revision` to supply an explicit corrected date range; the graph will re-check it rather than inventing a change.

## Tests and lint

```bash
uv run pytest
uv run ruff check .
```

## Project map

```text
src/tahoe_operator/
  chat.py             gateway-backed free-text plan + MCP execution
  api.py             FastAPI operator API and trace page
  config.py          environment and gateway settings
  graph.py           LangGraph control path and critic
  house_client.py    LangGraph-to-House-MCP client
  mcp_server.py      single MCP tool server
  services.py        deterministic compliance/calendar services
  storage.py         local SQLite property/event store
  corpus.py          Chroma vector index + gateway embeddings/fallback
data/
  corpus/             synthetic permit and house-rule documents
  fixtures/           fictional permit, weather, and calendar events
tests/                deterministic service, graph, and chat tests
.vscode/              interpreter, tasks, extensions, and launch config
```

## Gateway contract

Set `MODEL_GATEWAY_BASE_URL` to the OpenAI-compatible `/v1` endpoint you use now. `MODEL_GATEWAY_API_KEY` is read only from the environment. Free-text chat uses `MODEL_NAME` only to extract a typed, allow-listed plan. Embeddings use `EMBEDDING_MODEL` when configured. No provider URL or credential is assumed.

## Next steps after the first demo

1. Confirm the gateway's chat and embeddings routes/models and index the local corpus.
2. Add persistent human approval UI for calendar proposals and an audit view.
3. Add acceptance tests for permit expiry, occupancy, owner conflicts, stale sources, retries, and escalation.
4. Only then add pricing, turnover, guest desk, finance, and external read-only adapters.
