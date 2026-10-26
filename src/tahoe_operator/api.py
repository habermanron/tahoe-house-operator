import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from tahoe_operator.chat import GatewayNotConfigured, ShortTermChatMemory, run_chat
from tahoe_operator.graph import run_request
from tahoe_operator.house_client import HouseMCPClient
from tahoe_operator.schemas import ChatRequest, RunRequest

app = FastAPI(title="Tahoe House Operator MVP", version="0.1.0")
CLIENT = HouseMCPClient()
CHAT_MEMORY = ShortTermChatMemory()
STATIC_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "operator-api"}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return STATIC_PAGE.read_text(encoding="utf-8")


@app.post("/run")
async def run(body: RunRequest) -> dict[str, Any]:
    try:
        state = await run_request(body, CLIENT)
        return {
            "run_id": state.get("run_id"),
            "report_id": state.get("report_id"),
            "status": state.get("status"),
            "rounds": state.get("round", 1),
            "slices": state.get("slices", []),
            "critic": state.get("critic", {}),
            "tool_trace": state.get("tool_trace", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MVP run failed: {exc}") from exc


@app.post("/chat")
async def chat(body: ChatRequest) -> dict[str, Any]:
    try:
        return await run_chat(body, house=CLIENT, memory=CHAT_MEMORY)
    except GatewayNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Chat request failed: {exc}") from exc


@app.post("/run/stream")
async def run_stream(body: RunRequest) -> StreamingResponse:
    """Simple SSE-like event stream for the plain operator page."""

    async def events():
        event_id = str(uuid4())
        yield f"event: started\ndata: {json.dumps({'run_id': event_id})}\n\n"
        try:
            state = await run_request(body, CLIENT)
            yield f"event: result\ndata: {json.dumps(state, default=str)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/corpus/search")
async def corpus_search(query: str, limit: int = 4) -> dict[str, Any]:
    """Expose retrieval results for the operator's RAG inspection view."""
    try:
        return await CLIENT.call_tool("search_corpus", {"query": query, "limit": limit})
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"House MCP unavailable: {exc}") from exc


@app.get("/calendar")
async def calendar(start_date: str, end_date: str) -> dict[str, Any]:
    try:
        return await CLIENT.call_tool(
            "get_timeline", {"start_date": start_date, "end_date": end_date}
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"House MCP unavailable: {exc}") from exc
