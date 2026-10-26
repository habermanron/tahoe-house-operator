from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class OperatorRevision(BaseModel):
    """An explicit operator-supplied correction; the system never invents dates or counts."""

    start_date: date | None = None
    end_date: date | None = None
    guest_count: int | None = Field(default=None, ge=1)
    renter_age: int | None = Field(default=None, ge=18)
    vehicles: int | None = Field(default=None, ge=0)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    start_date: date | None = None
    end_date: date | None = None
    trash_date: date | None = None
    guest_count: int = Field(default=1, ge=1, le=30)
    renter_age: int | None = Field(default=None, ge=18, le=120)
    vehicles: int = Field(default=1, ge=0, le=20)
    proposed_action: str = "check_stay"
    operator_revision: OperatorRevision | None = None


class RunResult(BaseModel):
    run_id: str
    status: str
    rounds: int
    slices: list[dict[str, Any]]
    critic: dict[str, Any]
    tool_trace: list[dict[str, Any]]
    report_id: str


class ChatTask(BaseModel):
    kind: Literal[
        "occupancy",
        "availability",
        "stay_check",
        "trash_check",
        "parking_check",
        "rules_search",
        "feature_status",
    ]
    start_date: date | None = None
    end_date: date | None = None
    check_date: date | None = None
    guest_count: int | None = Field(default=None, ge=1, le=30)
    renter_age: int | None = Field(default=None, ge=18, le=120)
    vehicles: int | None = Field(default=None, ge=0, le=20)
    query: str | None = Field(default=None, max_length=500)


class ChatPlan(BaseModel):
    tasks: list[ChatTask] = Field(default_factory=list, max_length=6)
    clarifying_question: str | None = None
    unsupported_features: list[str] = Field(default_factory=list, max_length=6)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tasks: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    citations: list[dict[str, Any]]
