import json
from datetime import date, timedelta
from typing import Any

from tahoe_operator.config import get_settings
from tahoe_operator.storage import HouseStore


def _load_fixture(name: str) -> dict[str, Any]:
    path = get_settings().fixture_path / name
    return json.loads(path.read_text())


class ComplianceService:
    def get_permit(self) -> dict[str, Any]:
        return _load_fixture("permit.json")

    def occupancy_limit(self) -> dict[str, Any]:
        permit = self.get_permit()
        candidates = {
            "zone_limit": permit["zone_occupancy_limit"],
            "bedroom_limit": permit["bedroom_occupancy_limit"],
            "parking_limit": permit["paved_parking_limit"],
        }
        return {
            "limit": min(candidates.values()),
            "inputs": candidates,
            "source_id": "fixture-permit",
        }

    def check_stay(
        self,
        start_date: str,
        end_date: str,
        guest_count: int,
        renter_age: int | None,
        vehicles: int,
    ) -> dict[str, Any]:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        permit = self.get_permit()
        limit = self.occupancy_limit()["limit"]
        reasons: list[str] = []
        if end <= start:
            reasons.append("end_date_must_follow_start_date")
        if date.today() > date.fromisoformat(permit["expires_on"]):
            reasons.append("permit_expired")
        if guest_count > limit:
            reasons.append("occupancy_limit_exceeded")
        if renter_age is None:
            reasons.append("renter_age_required_for_check")
        elif renter_age < permit["minimum_renter_age"]:
            reasons.append("minimum_renter_age_not_met")
        if vehicles > permit["paved_parking_limit"]:
            reasons.append("vehicle_limit_exceeded")
        return {
            "status": "fail" if reasons else "pass",
            "reasons": reasons,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "guest_count": guest_count,
            "occupancy_limit": limit,
            "vehicles": vehicles,
            "source_id": "fixture-permit",
        }

    def check_trash(self, check_date: str) -> dict[str, Any]:
        target = date.fromisoformat(check_date)
        permit = self.get_permit()
        forecast = _load_fixture("weather.json").get(target.isoformat(), {})
        pickup_weekday = permit["trash_pickup_weekday"]
        is_pickup_day = target.weekday() == pickup_weekday
        bear_box_secured = permit["bear_box_secured"]
        storm_risk = forecast.get("storm_risk", "unknown")
        reasons: list[str] = []
        if is_pickup_day and not bear_box_secured:
            reasons.append("bear_box_not_secured_on_pickup_day")
        if is_pickup_day and storm_risk in {"high", "severe"} and not bear_box_secured:
            reasons.append("storm_and_unsecured_bear_box")
        return {
            "status": "fail" if reasons else "pass",
            "date": target.isoformat(),
            "pickup_day": is_pickup_day,
            "bear_box_secured": bear_box_secured,
            "storm_risk": storm_risk,
            "action": "secure bear box before pickup" if reasons else "no action required",
            "reasons": reasons,
            "source_id": "fixture-house-rules",
        }

    def check_parking(self, check_date: str, vehicles: int) -> dict[str, Any]:
        permit = self.get_permit()
        limit = permit["paved_parking_limit"]
        return {
            "status": "pass" if vehicles <= limit else "fail",
            "date": date.fromisoformat(check_date).isoformat(),
            "vehicles": vehicles,
            "parking_limit": limit,
            "source_id": "fixture-house-rules",
        }


class CalendarService:
    def __init__(self, store: HouseStore | None = None):
        self.store = store or HouseStore()

    def get_timeline(self, start_date: str, end_date: str) -> dict[str, Any]:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        return {
            "events": self.store.list_events(start, end),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }

    def find_gaps(self, start_date: str, end_date: str) -> dict[str, Any]:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        events = self.store.list_events(start, end)
        occupied = set()
        for event in events:
            if event["event_type"] in {"booking", "owner_block", "hold", "maintenance"}:
                cursor = max(start, date.fromisoformat(event["start_date"]))
                stop = min(end, date.fromisoformat(event["end_date"]))
                while cursor < stop:
                    occupied.add(cursor)
                    cursor += timedelta(days=1)
        gaps = []
        cursor = start
        while cursor < end:
            if cursor not in occupied:
                gaps.append(cursor.isoformat())
            cursor += timedelta(days=1)
        return {"open_nights": gaps, "events_considered": len(events)}

    def hold_dates(self, start_date: str, end_date: str, approved: bool = False) -> dict[str, Any]:
        return self.store.create_event_proposal(
            "hold", date.fromisoformat(start_date), date.fromisoformat(end_date), {}, approved
        )

    def block_owner(self, start_date: str, end_date: str, approved: bool = False) -> dict[str, Any]:
        return self.store.create_event_proposal(
            "owner_block",
            date.fromisoformat(start_date),
            date.fromisoformat(end_date),
            {},
            approved,
        )


def log_incident(category: str, description: str, source_rule_id: str) -> dict[str, Any]:
    return HouseStore().log_incident(category, description, source_rule_id)
