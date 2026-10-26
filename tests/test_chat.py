from tahoe_operator.chat import ShortTermChatMemory, run_chat
from tahoe_operator.schemas import ChatPlan, ChatRequest, ChatTask


class FakePlanner:
    def __init__(self, plan):
        self.result = plan
        self.histories = []

    async def plan(self, message, history):
        self.histories.append(list(history))
        return self.result


class FakeHouseMCPClient:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "get_permit":
            return {"permit_id": "SYNTHETIC-001", "source_id": "fixture-permit"}
        if name == "occupancy_limit":
            return {"limit": 2, "inputs": {"bedroom_limit": 6, "zone_limit": 6, "parking_limit": 2}}
        raise AssertionError(f"Unexpected MCP tool: {name}")


async def test_chat_uses_gateway_plan_but_tool_results_for_occupancy():
    planner = FakePlanner(ChatPlan(tasks=[ChatTask(kind="occupancy")]))
    house = FakeHouseMCPClient()
    memory = ShortTermChatMemory()
    result = await run_chat(
        ChatRequest(message="How many guests can stay?"), house, planner, memory
    )

    assert "limit is 2 guests" in result["answer"]
    assert [name for name, _ in house.calls] == ["get_permit", "occupancy_limit"]
    assert result["citations"][0]["source_id"] == "fixture-permit"


async def test_chat_retains_only_short_term_conversation_context():
    memory = ShortTermChatMemory()
    planner = FakePlanner(ChatPlan(clarifying_question="Which dates do you mean?"))
    request = ChatRequest(message="Is it available?", session_id="test-session")

    await run_chat(request, FakeHouseMCPClient(), planner, memory)
    await run_chat(
        ChatRequest(message="Next weekend", session_id="test-session"),
        FakeHouseMCPClient(),
        planner,
        memory,
    )

    assert planner.histories[0] == []
    assert planner.histories[1][0]["role"] == "user"
    assert len(memory.recent("test-session")) == 4
