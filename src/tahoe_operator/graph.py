import operator
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from tahoe_operator.agents.calendar import CalendarSpecialist
from tahoe_operator.agents.compliance import ComplianceSpecialist
from tahoe_operator.agents.knowledge import KnowledgeSpecialist
from tahoe_operator.house_client import HouseMCPClient
from tahoe_operator.schemas import RunRequest


class OperatorState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    round: int
    revision_applied: bool
    selected_slices: list[str]
    compliance_result: dict[str, Any]
    calendar_result: dict[str, Any]
    knowledge_result: dict[str, Any]
    slices: list[dict[str, Any]]
    failed_slices: list[str]
    tool_trace: Annotated[list[dict[str, Any]], operator.add]
    critic: dict[str, Any]
    status: str
    report_id: str


def build_graph(client: HouseMCPClient | None = None):
    """Build the supervisor graph with parallel, tool-scoped domain specialists."""
    house = client or HouseMCPClient()
    compliance_agent = ComplianceSpecialist(house)
    calendar_agent = CalendarSpecialist(house)
    knowledge_agent = KnowledgeSpecialist(house)

    async def supervisor(state: OperatorState) -> dict[str, Any]:
        request = dict(state["request"])
        selected = []
        if (request.get("start_date") and request.get("end_date")) or request.get("trash_date"):
            selected.extend(["compliance", "calendar"])
        if request.get("prompt"):
            selected.append("knowledge")
        return {
            "run_id": state.get("run_id") or str(uuid4()),
            "round": state.get("round", 1),
            "request": request,
            "selected_slices": selected,
            "compliance_result": {"name": "compliance", "status": "skipped", "results": []},
            "calendar_result": {"name": "calendar", "status": "skipped", "results": []},
            "knowledge_result": {"name": "knowledge", "status": "skipped", "results": []},
        }

    async def compliance_specialist(state: OperatorState) -> dict[str, Any]:
        result, trace = await compliance_agent.inspect_request(state["request"])
        return {"compliance_result": result, "tool_trace": trace}

    async def calendar_specialist(state: OperatorState) -> dict[str, Any]:
        result, trace = await calendar_agent.inspect_request(state["request"])
        return {"calendar_result": result, "tool_trace": trace}

    async def knowledge_specialist(state: OperatorState) -> dict[str, Any]:
        result, trace = await knowledge_agent.inspect_request(state["request"])
        return {"knowledge_result": result, "tool_trace": trace}

    async def critic(state: OperatorState) -> dict[str, Any]:
        slices = [
            state.get("compliance_result", {}),
            state.get("calendar_result", {}),
            state.get("knowledge_result", {}),
        ]
        slices = [item for item in slices if item.get("status") != "skipped"]
        failed: list[str] = []
        reasons: list[str] = []
        for item in slices:
            for result in item.get("results", []):
                kind = result.get("kind")
                if kind == "stay_check":
                    stay, parking = result.get("stay", {}), result.get("parking", {})
                    if stay.get("status") != "pass":
                        failed.append("compliance")
                        reasons.extend(stay.get("reasons", []))
                    if parking.get("status") != "pass":
                        failed.append("compliance")
                        reasons.append("parking_check_failed")
                elif kind in {"trash_check", "parking_check"}:
                    tool_result = result.get("result", {})
                    if tool_result.get("status") == "fail":
                        failed.append("compliance")
                        reasons.extend(tool_result.get("reasons", []))
                elif kind in {"availability", "stay_calendar"} and result.get("conflicts"):
                    failed.append("calendar")
                    reasons.append("requested_dates_overlap_calendar_event")
                if kind == "rules_search" and not result.get("result", {}).get("chunks"):
                    failed.append("knowledge")
                    reasons.append("required_corpus_evidence_not_found")
        failed = list(dict.fromkeys(failed))
        verdict = {
            "status": "fail" if failed else "pass",
            "failed_slices": failed,
            "reasons": list(dict.fromkeys(reasons)),
            "round": state.get("round", 1),
            "tools_called": False,
            "specialists_reviewed": [item["name"] for item in slices],
        }
        return {
            "slices": slices,
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
        }

    async def reporter(state: OperatorState) -> dict[str, Any]:
        return {"report_id": str(uuid4())}

    builder = StateGraph(OperatorState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("compliance_specialist", compliance_specialist)
    builder.add_node("calendar_specialist", calendar_specialist)
    builder.add_node("knowledge_specialist", knowledge_specialist)
    builder.add_node("critic", critic)
    builder.add_node("revise", revise)
    builder.add_node("reporter", reporter)
    builder.add_edge(START, "supervisor")

    def dispatch_specialists(state: OperatorState) -> list[Send]:
        sends = []
        request = state["request"]
        for name, node in (
            ("compliance", "compliance_specialist"),
            ("calendar", "calendar_specialist"),
            ("knowledge", "knowledge_specialist"),
        ):
            if name in state.get("selected_slices", []):
                sends.append(Send(node, {"request": request}))
        if not sends:
            sends.append(Send("critic", {}))
        return sends

    builder.add_conditional_edges("supervisor", dispatch_specialists)
    builder.add_edge("compliance_specialist", "critic")
    builder.add_edge("calendar_specialist", "critic")
    builder.add_edge("knowledge_specialist", "critic")

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
    builder.add_edge("revise", "supervisor")
    builder.add_edge("reporter", END)
    return builder.compile()


async def run_request(request: RunRequest, client: HouseMCPClient | None = None) -> dict[str, Any]:
    graph = build_graph(client)
    initial: OperatorState = {
        "run_id": str(uuid4()),
        "request": request.model_dump(mode="json"),
        "tool_trace": [],
    }
    return await graph.ainvoke(initial)
