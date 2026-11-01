from typing import Any

from tahoe_operator.schemas import ChatTask


class ScopeSpecialist:
    """Explains the supported MVP boundary without calling operational tools."""

    name = "scope"

    @staticmethod
    async def execute_task(task: ChatTask) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return {"kind": task.kind, "specialist": "scope"}, []
