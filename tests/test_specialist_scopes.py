import pytest

from tahoe_operator.agents.calendar import CalendarSpecialist
from tahoe_operator.agents.compliance import ComplianceSpecialist
from tahoe_operator.agents.knowledge import KnowledgeSpecialist


class NoCallHouseClient:
    async def call_tool(self, name, arguments):
        raise AssertionError("Disallowed tool must be rejected before reaching MCP")


@pytest.mark.asyncio
async def test_specialists_enforce_independent_tool_scopes():
    house = NoCallHouseClient()
    with pytest.raises(PermissionError):
        await ComplianceSpecialist(house).call_tool("get_timeline", {})
    with pytest.raises(PermissionError):
        await CalendarSpecialist(house).call_tool("search_corpus", {})
    with pytest.raises(PermissionError):
        await KnowledgeSpecialist(house).call_tool("hold_dates", {})
