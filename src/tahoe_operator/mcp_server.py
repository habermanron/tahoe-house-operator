from typing import Any

from mcp.server.fastmcp import FastMCP

from tahoe_operator.config import get_settings
from tahoe_operator.corpus import CorpusService
from tahoe_operator.services import CalendarService, ComplianceService
from tahoe_operator.services import log_incident as record_incident

settings = get_settings()
mcp = FastMCP("Tahoe House MCP", host=settings.mcp_host, port=settings.mcp_port)
compliance = ComplianceService()
calendar = CalendarService()
corpus = CorpusService()


@mcp.tool()
def get_permit() -> dict[str, Any]:
    """Return the synthetic permit fixture and its source identifier."""
    return compliance.get_permit()


@mcp.tool()
def occupancy_limit() -> dict[str, Any]:
    """Compute the permitted occupancy using the lesser configured limit."""
    return compliance.occupancy_limit()


@mcp.tool()
def check_stay(
    start_date: str,
    end_date: str,
    guest_count: int,
    renter_age: int | None = None,
    vehicles: int = 1,
) -> dict[str, Any]:
    """Check a proposed stay against the synthetic permit constraints."""
    return compliance.check_stay(start_date, end_date, guest_count, renter_age, vehicles)


@mcp.tool()
def check_trash(check_date: str) -> dict[str, Any]:
    """Check pickup-day, bear-box, and fixture storm-risk conditions."""
    return compliance.check_trash(check_date)


@mcp.tool()
def check_parking(check_date: str, vehicles: int) -> dict[str, Any]:
    """Check vehicles against the paved-parking fixture limit."""
    return compliance.check_parking(check_date, vehicles)


@mcp.tool()
def log_incident(category: str, description: str, source_rule_id: str) -> dict[str, Any]:
    """Record an operator-confirmed synthetic incident against a rule source."""
    return record_incident(category, description, source_rule_id)


@mcp.tool()
def get_timeline(start_date: str, end_date: str) -> dict[str, Any]:
    """Return typed calendar events in a half-open date range."""
    return calendar.get_timeline(start_date, end_date)


@mcp.tool()
def find_gaps(start_date: str, end_date: str) -> dict[str, Any]:
    """Return open nights in a date range based on typed calendar events."""
    return calendar.find_gaps(start_date, end_date)


@mcp.tool()
def hold_dates(start_date: str, end_date: str, approved: bool = False) -> dict[str, Any]:
    """Propose a hold; it is not written unless the operator supplies approval."""
    return calendar.hold_dates(start_date, end_date, approved)


@mcp.tool()
def block_owner(start_date: str, end_date: str, approved: bool = False) -> dict[str, Any]:
    """Propose an owner block; it is not written unless the operator supplies approval."""
    return calendar.block_owner(start_date, end_date, approved)


@mcp.tool()
async def search_corpus(query: str, limit: int = 4) -> dict[str, Any]:
    """Search the local policy corpus and return passages with source metadata."""
    result = await corpus.search(query, limit)
    if result["status"] == "ok":
        result["citations"] = [
            {
                "source_id": chunk["source_id"],
                "title": chunk["title"],
                "version": chunk["version"],
                "effective_date": chunk["effective_date"],
            }
            for chunk in result["chunks"]
        ]
    return result


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
