from typing import Any

from tahoe_operator.house_client import HouseMCPClient


class ToolSpecialist:
    """Tool-scoped specialist worker; no model-selected tool names are accepted."""

    name = "specialist"
    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, house: HouseMCPClient):
        self.house = house

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if name not in self.allowed_tools:
            raise PermissionError(f"{self.name} is not allowed to call {name}")
        result = await self.house.call_tool(name, arguments)
        trace = {
            "specialist": self.name,
            "tool": name,
            "arguments": arguments,
            "result": result,
        }
        return result, trace
