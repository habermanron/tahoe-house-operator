from datetime import date
from typing import Any

from tahoe_operator.agents.base import ToolSpecialist
from tahoe_operator.schemas import ChatTask


class CalendarSpecialist(ToolSpecialist):
    """Checks availability and conflicts; it cannot modify calendar events."""

    name = "calendar"
    allowed_tools = frozenset({"get_timeline", "find_gaps"})
    blocking_events = {"owner_block", "booking", "hold", "maintenance"}

    @staticmethod
    def _conflicts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            event
            for event in events
            if event.get("event_type") in CalendarSpecialist.blocking_events
        ]

    async def execute_task(self, task: ChatTask) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if task.kind not in {"availability", "stay_check"}:
            return {"kind": task.kind, "specialist": self.name, "skipped": True}, []
        if not task.start_date or not task.end_date:
            return {
                "kind": "stay_calendar" if task.kind == "stay_check" else task.kind,
                "specialist": self.name,
                "needs_clarification": True,
            }, []

        start, end = task.start_date.isoformat(), task.end_date.isoformat()
        args = {"start_date": start, "end_date": end}
        timeline, timeline_trace = await self.call_tool("get_timeline", args)
        events = timeline.get("events", [])
        conflicts = self._conflicts(events)
        trace = [timeline_trace]
        evidence: dict[str, Any] = {
            "kind": "stay_calendar" if task.kind == "stay_check" else "availability",
            "specialist": self.name,
            "events": events,
            "conflicts": conflicts,
            "start_date": start,
            "end_date": end,
        }
        if task.kind == "availability":
            gaps, gaps_trace = await self.call_tool("find_gaps", args)
            evidence["gaps"] = gaps
            trace.append(gaps_trace)
        return evidence, trace

    async def inspect_request(
        self, request: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        start, end = request.get("start_date"), request.get("end_date")
        if not start and not request.get("trash_date"):
            return {"name": self.name, "status": "skipped", "results": []}, []
        if not start:
            start = request.get("trash_date") or date.today().isoformat()
        if not end:
            end = date.fromordinal(date.fromisoformat(start).toordinal() + 1).isoformat()
        task = ChatTask(kind="availability", start_date=start, end_date=end)
        evidence, trace = await self.execute_task(task)
        return {"name": self.name, "status": "complete", "results": [evidence]}, trace
