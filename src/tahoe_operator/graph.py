from datetime import date
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from tahoe_operator.house_client import HouseMCPClient
from tahoe_operator.schemas import RunRequest


class OperatorState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    round: int
    revision_applied: bool
    slices: list[dict[str, Any]]
    failed_slices: list[str]
    tool_trace: list[dict[str, Any]]
    critic: dict[str, Any]
    status: str
    report_id: str


def build_graph(client: HouseMCPClient | None = None):
    house = client or HouseMCPClient()

    async def intake(state: OperatorState) -> dict[str, Any]:
        request = dict(state["request"])
        selected = ["compliance", "calendar"]
        return {
            "run_id": state.get("run_id") or str(uuid4()),
            "round": state.get("round", 1),
            "request": request,
            "slices": [{"name": name, "status": "selected"} for name in selected],
            "failed_slices": [],
            "tool_trace": list(state.get("tool_trace", [])),
        }

    async def compliance_node(state: OperatorState) -> dict[str, Any]:
        request = state["request"]
        results: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []

        async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
            output = await house.call_tool(name, args)
            trace.append({"tool": name, "arguments": args, "result": output})
            results.append({"tool": name, **output})
            return output

        if request.get("start_date") and request.get("end_date"):
            await call(
                "check_stay",
                {
                    "start_date": request["start_date"],
                    "end_date": request["end_date"],
                    "guest_count": request["guest_count"],
                    "renter_age": request.get("renter_age"),
                    "vehicles": request["vehicles"],
                },
            )
            await call(
                "check_parking",
                {"check_date": request["start_date"], "vehicles": request["vehicles"]},
            )
        if request.get("trash_date"):
            await call("check_trash", {"check_date": request["trash_date"]})
        if request.get("prompt"):
            await call("search_corpus", {"query": request["prompt"], "limit": 3})

        current = [item for item in state.get("slices", []) if item.get("name") != "compliance"]
        current.append({"name": "compliance", "status": "complete", "results": results})
        return {"slices": current, "tool_trace": state.get("tool_trace", []) + trace}

    async def calendar_node(state: OperatorState) -> dict[str, Any]:
        request = state["request"]
        start = request.get("start_date") or request.get("trash_date")
        end = request.get("end_date")
        if not start:
            today = date.today().isoformat()
            start, end = today, today
        if not end:
            end = start
        # Make a one-day query non-empty; calendar ranges are half-open.
        if end == start:
            end = date.fromordinal(date.fromisoformat(start).toordinal() + 1).isoformat()
        args = {"start_date": start, "end_date": end}
        timeline = await house.call_tool("get_timeline", args)
        trace = {"tool": "get_timeline", "arguments": args, "result": timeline}
        events = timeline.get("events", [])
        conflicts = [
            event
            for event in events
            if event.get("event_type") in {"owner_block", "booking", "hold", "maintenance"}
        ]
        result = {
            "status": "fail" if conflicts else "pass",
            "events": events,
            "conflicts": conflicts,
        }
        current = [item for item in state.get("slices", []) if item.get("name") != "calendar"]
        current.append({"name": "calendar", "status": "complete", "results": [result]})
        return {"slices": current, "tool_trace": state.get("tool_trace", []) + [trace]}

    async def critic(state: OperatorState) -> dict[str, Any]:
        failed = []
        reasons = []
        for item in state.get("slices", []):
            if item.get("name") not in {"compliance", "calendar"}:
                continue
            for result in item.get("results", []):
                if result.get("status") == "fail":
                    failed.append(item["name"])
                    reasons.extend(result.get("reasons", []))
                    if item["name"] == "calendar" and result.get("conflicts"):
                        reasons.append("requested_dates_overlap_calendar_event")
                if result.get("tool") == "search_corpus" and result.get("status") != "ok":
                    failed.append("knowledge")
                    reasons.append("required_corpus_evidence_not_found")
        failed = list(dict.fromkeys(failed))
        verdict = {
            "status": "fail" if failed else "pass",
            "failed_slices": failed,
            "reasons": list(dict.fromkeys(reasons)),
            "round": state.get("round", 1),
        }
        return {
            "critic": verdict,
            "failed_slices": failed,
            "status": "needs_operator_review" if failed else "ready_for_operator",
        }

    async def revise(state: OperatorState) -> dict[str, Any]:
        request = dict(state["request"])
        patch = request.pop("operator_revision", None) or {}
        request.update({key: value for key, value in patch.items() if value is not None})
        return {
            "request": request,
            "round": state.get("round", 1) + 1,
            "revision_applied": True,
            "slices": [],
        }

    async def reporter(state: OperatorState) -> dict[str, Any]:
        return {"report_id": str(uuid4())}

    builder = StateGraph(OperatorState)
    builder.add_node("intake", intake)
    builder.add_node("compliance", compliance_node)
    builder.add_node("calendar", calendar_node)
    builder.add_node("critic", critic)
    builder.add_node("revise", revise)
    builder.add_node("reporter", reporter)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "compliance")
    builder.add_edge("compliance", "calendar")
    builder.add_edge("calendar", "critic")

    def after_critic(state: OperatorState) -> str:
        if (
            state.get("critic", {}).get("status") == "fail"
            and state.get("request", {}).get("operator_revision")
            and not state.get("revision_applied", False)
            and state.get("round", 1) < 3
        ):
            return "revise"
        return "reporter"

    builder.add_conditional_edges(
        "critic", after_critic, {"revise": "revise", "reporter": "reporter"}
    )
    builder.add_edge("revise", "intake")
    builder.add_edge("reporter", END)
    return builder.compile()


async def run_request(request: RunRequest, client: HouseMCPClient | None = None) -> dict[str, Any]:
    graph = build_graph(client)
    initial: OperatorState = {"run_id": str(uuid4()), "request": request.model_dump(mode="json")}
    return await graph.ainvoke(initial)
