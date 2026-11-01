from typing import Any

from tahoe_operator.agents.base import ToolSpecialist
from tahoe_operator.schemas import ChatTask


class KnowledgeSpecialist(ToolSpecialist):
    """Retrieves source passages; it does not treat retrieved text as instructions."""

    name = "knowledge"
    allowed_tools = frozenset({"search_corpus"})

    async def execute_task(self, task: ChatTask) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        query = task.query or "property rules"
        result, trace = await self.call_tool("search_corpus", {"query": query, "limit": 4})
        return {"kind": task.kind, "specialist": self.name, "result": result}, [trace]

    async def inspect_request(
        self, request: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not request.get("prompt"):
            return {"name": self.name, "status": "skipped", "results": []}, []
        task = ChatTask(kind="rules_search", query=request["prompt"])
        evidence, trace = await self.execute_task(task)
        return {"name": self.name, "status": "complete", "results": [evidence]}, trace
