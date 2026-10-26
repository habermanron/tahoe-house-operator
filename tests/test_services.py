from datetime import date

from tahoe_operator.services import CalendarService, ComplianceService
from tahoe_operator.storage import HouseStore


def test_occupancy_uses_least_configured_limit():
    result = ComplianceService().occupancy_limit()
    assert result["limit"] == 2
    assert result["inputs"]["parking_limit"] == 2


def test_stay_rejects_missing_renter_age():
    result = ComplianceService().check_stay("2026-11-10", "2026-11-11", 2, None, 1)
    assert result["status"] == "fail"
    assert "renter_age_required_for_check" in result["reasons"]


def test_trash_flags_unsecured_box_on_fixture_pickup_day():
    result = ComplianceService().check_trash("2026-11-13")
    assert result["status"] == "fail"
    assert result["storm_risk"] == "high"


def test_hold_needs_operator_approval(tmp_path):
    store = HouseStore(tmp_path / "test.sqlite")
    result = store.create_event_proposal(
        "hold", date(2026, 12, 1), date(2026, 12, 2), {}, approved=False
    )
    assert result["status"] == "needs_approval"
    assert result["calendar_changed"] is False


def test_calendar_returns_owner_block_conflict():
    result = CalendarService().get_timeline("2026-11-14", "2026-11-15")
    assert any(event["event_type"] == "owner_block" for event in result["events"])
