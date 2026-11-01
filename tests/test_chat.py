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
        if name == "check_stay":
            return {"status": "pass", "reasons": [], "source_id": "fixture-permit"}
        if name == "check_parking":
            return {"status": "pass", "source_id": "fixture-house-rules"}
        if name == "get_timeline":
            return {"events": []}
        if name == "find_gaps":
            return {"open_nights": [arguments["start_date"]]}
        if name == "search_corpus":
            return {
                "status": "ok",
                "chunks": [{"text": "Synthetic rule excerpt", "source_id": "fixture-house-rules"}],
                "citations": [{"source_id": "fixture-house-rules", "title": "House rules"}],
            }
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
    assert {call["specialist"] for call in result["tool_trace"]} == {"compliance"}


async def test_supervisor_fans_mixed_question_out_to_scoped_specialists():
    planner = FakePlanner(
        ChatPlan(
            tasks=[
                ChatTask(
                    kind="stay_check",
                    start_date="2026-11-16",
                    end_date="2026-11-17",
                    guest_count=2,
                    renter_age=35,
                    vehicles=1,
                ),
                ChatTask(kind="rules_search", query="parking rules"),
            ]
        )
    )
    house = FakeHouseMCPClient()
    result = await run_chat(
        ChatRequest(message="Can I stay then, and what are the parking rules?"),
        house,
        planner,
        ShortTermChatMemory(),
    )

    specialists = {call["specialist"] for call in result["tool_trace"]}
    assert specialists == {"compliance", "calendar", "knowledge"}
    assert result["critic"]["status"] == "pass"
    assert any(call[0] == "search_corpus" for call in house.calls)


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
