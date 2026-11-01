from tahoe_operator.graph import run_request
from tahoe_operator.schemas import RunRequest


class FakeHouseMCPClient:
    async def call_tool(self, name, arguments):
        if name == "check_stay":
            return {"status": "pass", "reasons": [], "source_id": "fixture-permit"}
        if name == "check_parking":
            return {"status": "pass", "source_id": "fixture-house-rules"}
        if name == "check_trash":
            return {"status": "pass", "reasons": [], "source_id": "fixture-house-rules"}
        if name == "search_corpus":
            return {"status": "ok", "chunks": [{"source_id": "fixture-house-rules"}]}
        if name == "get_timeline":
            conflict = arguments["start_date"] == "2026-11-14"
            events = (
                [
                    {
                        "id": "owner-block",
                        "event_type": "owner_block",
                        "start_date": "2026-11-14",
                        "end_date": "2026-11-16",
                    }
                ]
                if conflict
                else []
            )
            return {"events": events}
        if name == "find_gaps":
            return {"open_nights": [arguments["start_date"]], "events_considered": 0}
        raise AssertionError(f"Unexpected MCP tool: {name}")


async def test_operator_revision_rechecks_a_failed_owner_conflict():
    request = RunRequest(
        prompt="Check an owner stay and calendar conflict",
        start_date="2026-11-14",
        end_date="2026-11-15",
        guest_count=2,
        renter_age=35,
        vehicles=1,
        operator_revision={"start_date": "2026-11-16", "end_date": "2026-11-17"},
    )
    result = await run_request(request, FakeHouseMCPClient())
    assert result["round"] == 2
    assert result["critic"]["status"] == "pass"
    assert result["request"]["start_date"] == "2026-11-16"
