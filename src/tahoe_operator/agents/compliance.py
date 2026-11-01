from typing import Any

from tahoe_operator.agents.base import ToolSpecialist
from tahoe_operator.schemas import ChatTask


class ComplianceSpecialist(ToolSpecialist):
    """Checks stays, occupancy, parking, and trash; never changes property state."""

    name = "compliance"
    allowed_tools = frozenset(
        {"get_permit", "occupancy_limit", "check_stay", "check_trash", "check_parking"}
    )

    async def execute_task(self, task: ChatTask) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = []
        if task.kind == "occupancy":
            permit, permit_trace = await self.call_tool("get_permit", {})
            limit, limit_trace = await self.call_tool("occupancy_limit", {})
            trace.extend([permit_trace, limit_trace])
            return {
                "kind": task.kind,
                "specialist": self.name,
                "permit": permit,
                "occupancy": limit,
            }, trace

        if task.kind == "stay_check":
            if (
                not task.start_date
                or not task.end_date
                or task.guest_count is None
                or task.renter_age is None
                or task.vehicles is None
            ):
                return {
                    "kind": task.kind,
                    "specialist": self.name,
                    "needs_clarification": True,
                }, trace
            start, end = task.start_date.isoformat(), task.end_date.isoformat()
            stay, stay_trace = await self.call_tool(
                "check_stay",
                {
                    "start_date": start,
                    "end_date": end,
                    "guest_count": task.guest_count,
                    "renter_age": task.renter_age,
                    "vehicles": task.vehicles,
                },
            )
            parking, parking_trace = await self.call_tool(
                "check_parking",
                {
                    "check_date": start,
                    "vehicles": task.vehicles,
                },
            )
            trace.extend([stay_trace, parking_trace])
            return {
                "kind": task.kind,
                "specialist": self.name,
                "start_date": start,
                "end_date": end,
                "stay": stay,
                "parking": parking,
            }, trace

        if task.kind == "trash_check":
            if not task.check_date:
                return {
                    "kind": task.kind,
                    "specialist": self.name,
                    "needs_clarification": True,
                }, trace
            result, item_trace = await self.call_tool(
                "check_trash", {"check_date": task.check_date.isoformat()}
            )
            trace.append(item_trace)
            return {"kind": task.kind, "specialist": self.name, "result": result}, trace

        if task.kind == "parking_check":
            if not task.check_date or task.vehicles is None:
                return {
                    "kind": task.kind,
                    "specialist": self.name,
                    "needs_clarification": True,
                }, trace
            result, item_trace = await self.call_tool(
                "check_parking",
                {"check_date": task.check_date.isoformat(), "vehicles": task.vehicles},
            )
            trace.append(item_trace)
            return {"kind": task.kind, "specialist": self.name, "result": result}, trace

        return {"kind": task.kind, "specialist": self.name, "skipped": True}, trace

    async def inspect_request(
        self, request: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        tasks = []
        if request.get("start_date") and request.get("end_date"):
            tasks.append(
                ChatTask(
                    kind="stay_check",
                    start_date=request["start_date"],
                    end_date=request["end_date"],
                    guest_count=request.get("guest_count", 1),
                    renter_age=request.get("renter_age"),
                    vehicles=request.get("vehicles", 0),
                )
            )
        if request.get("trash_date"):
            tasks.append(ChatTask(kind="trash_check", check_date=request["trash_date"]))
        results, trace = [], []
        for task in tasks:
            evidence, task_trace = await self.execute_task(task)
            results.append(evidence)
            trace.extend(task_trace)
        return {
            "name": self.name,
            "status": "complete" if tasks else "skipped",
            "results": results,
        }, trace
