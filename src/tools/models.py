"""
Shared data models and helper utilities for MCP tools.
"""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator


class FactType(str, Enum):
    """Canonical fact types exchanged between MCP and clients."""

    CONSENT = "consent"
    AGE = "age"
    DEVICE = "device"
    COMMITMENT_HOURS = "commitment_hours"
    COMMITMENT_MONTHS = "commitment_months"
    PREFERRED_DAY = "preferred_day"
    PREFERRED_TIME = "preferred_time"
    PREFERRED_SLOT = "preferred_slot"
    ORIENTATION_SLOT = "orientation_slot"
    ORIENTATION_READY = "orientation_ready"
    FAQ_ANSWER = "faq_answer"


class Action(BaseModel):
    """Instruction for the client to trigger additional behaviour."""

    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    """Single normalized fact extracted from a volunteer turn."""

    type: FactType
    value: Any
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: Optional[str] = None

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: Any, info: Dict[str, Any]) -> Any:
        fact_type: FactType = info.data.get("type")  # type: ignore[attr-defined]
        return normalize_fact_value(fact_type, v)


class MCPResponse(BaseModel):
    """
    Standard response payload returned by MCP tools.

    Attributes:
        reply: WhatsApp-ready message for the volunteer.
        next_state: Conversation state to transition into (or "SAME").
        facts: Structured facts extracted during the turn.
        pending_questions: Follow-up prompts required to complete the state.
        actions: Optional automation instructions (e.g., schedule orientation).
        debug: Diagnostics for logging/observability (not sent to volunteers).
    """

    reply: str
    next_state: str
    facts: Optional[List[Fact]] = None
    pending_questions: Optional[List[str]] = None
    actions: Optional[List[Action]] = None
    debug: Optional[Dict[str, Any]] = None

    @field_validator("next_state")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper() if v else "SAME"

    def ensure_defaults(self) -> "MCPResponse":
        """Ensure optional containers are always lists for downstream convenience."""
        if self.facts is None:
            self.facts = []
        if self.pending_questions is None:
            self.pending_questions = []
        if self.actions is None:
            self.actions = []
        return self


class StateEvaluation(BaseModel):
    """Outcome of evaluating facts against state requirements."""

    next_state: str
    satisfied: List[FactType] = Field(default_factory=list)
    missing: List[FactType] = Field(default_factory=list)
    pending_questions: List[str] = Field(default_factory=list)


class Slot(BaseModel):
    """Time slot model - shared across all time-related tools."""

    start_iso: str
    end_iso: str
    label: str
    confidence: float = 0.8


_WEEKDAY_MAP = {
    "monday": "Mon",
    "mon": "Mon",
    "tuesday": "Tue",
    "tue": "Tue",
    "tues": "Tue",
    "wednesday": "Wed",
    "wed": "Wed",
    "thursday": "Thu",
    "thu": "Thu",
    "thur": "Thu",
    "thurs": "Thu",
    "friday": "Fri",
    "fri": "Fri",
}


def _normalize_consent(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, dict):
        value = value.get("value") or value.get("status")
    if isinstance(value, bool):
        return "yes" if value else "no"
    val = str(value).strip().lower()
    if val in {"yes", "y", "sure", "ok", "okay"}:
        return "yes"
    if val in {"no", "n", "nope"}:
        return "no"
    if val in {"defer", "maybe", "later"}:
        return "defer"
    return "unknown"


def _normalize_age(value: Any) -> Dict[str, Optional[Any]]:
    result: Dict[str, Optional[Any]] = {"years": None, "is_18_plus": None}
    if value is None:
        return result
    if isinstance(value, dict):
        if "years" in value and value["years"] is not None:
            try:
                result["years"] = int(value["years"])
            except (TypeError, ValueError):
                result["years"] = None
        if "is_18_plus" in value and value["is_18_plus"] is not None:
            result["is_18_plus"] = bool(value["is_18_plus"])
        if result["years"] is not None and result["is_18_plus"] is None:
            result["is_18_plus"] = result["years"] >= 18
        return result
    if isinstance(value, (int, float)):
        result["years"] = int(value)
    else:
        digits = re.findall(r"\d+", str(value))
        result["years"] = int(digits[0]) if digits else None
    if isinstance(value, bool):
        result["is_18_plus"] = value
    if result["years"] is not None and result["is_18_plus"] is None:
        result["is_18_plus"] = result["years"] >= 18
    return result


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"yes", "true", "has", "y", "ok", "okay"}:
        return True
    if val in {"no", "false", "none", "n"}:
        return False
    return None


def _normalize_device(value: Any) -> Dict[str, Optional[Any]]:
    result: Dict[str, Optional[Any]] = {"has_device": None, "device_type": None}
    if isinstance(value, dict):
        if "has_device" in value:
            has_device = value.get("has_device")
            result["has_device"] = bool(has_device) if has_device is not None else None
        if "device_type" in value and value["device_type"]:
            result["device_type"] = str(value["device_type"]).lower()
        if "device_ok" in value and value["device_ok"] is not None:
            result["has_device"] = bool(value["device_ok"])
        return result
    normalized = _normalize_bool(value)
    if normalized is not None:
        result["has_device"] = normalized
    if isinstance(value, str):
        low = value.lower()
        for device_word in ("laptop", "desktop", "phone", "smartphone", "tablet"):
            if device_word in low:
                result["device_type"] = "smartphone" if device_word == "phone" else device_word
                break
    return result


def _normalize_hours(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def _normalize_months(value: Any) -> Optional[float]:
    return _normalize_hours(value)


def _normalize_day(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _WEEKDAY_MAP:
            return _WEEKDAY_MAP[key]
        return value.strip().title()
    return None


def _normalize_time_window(value: Any) -> Optional[Dict[str, str]]:
    if value is None:
        return None
    if isinstance(value, dict):
        start = value.get("start") or value.get("start_time")
        end = value.get("end") or value.get("end_time")
        if start and end:
            return {"start": _normalize_hhmm(start), "end": _normalize_hhmm(end)}
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return {"start": _normalize_hhmm(value[0]), "end": _normalize_hhmm(value[1])}
    if isinstance(value, str) and "-" in value:
        start, end = value.split("-", 1)
        return {"start": _normalize_hhmm(start), "end": _normalize_hhmm(end)}
    return None


def _normalize_slot(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Slot):
        return value.model_dump()
    if isinstance(value, dict):
        start = value.get("start_iso") or value.get("start") or value.get("startTime")
        end = value.get("end_iso") or value.get("end") or value.get("endTime")
        label = value.get("label") or value.get("summary")
        data = {}
        if start:
            data["start_iso"] = _normalize_iso(start)
        if end:
            data["end_iso"] = _normalize_iso(end)
        if label:
            data["label"] = str(label)
        if "confidence" in value:
            data["confidence"] = float(value["confidence"])
        return data or None
    return None


def _normalize_hhmm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        hour = int(value)
        return f"{hour:02d}:00"
    text = str(value).strip()
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", text, re.I)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).lower() if m.group(3) else None
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    if re.match(r"^\d{2}:\d{2}$", text):
        return text
    return text


def _normalize_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return text


_FACT_NORMALIZERS: Dict[FactType, Callable[[Any], Any]] = {
    FactType.CONSENT: _normalize_consent,
    FactType.AGE: _normalize_age,
    FactType.DEVICE: _normalize_device,
    FactType.COMMITMENT_HOURS: _normalize_hours,
    FactType.COMMITMENT_MONTHS: _normalize_months,
    FactType.PREFERRED_DAY: _normalize_day,
    FactType.PREFERRED_TIME: _normalize_time_window,
    FactType.PREFERRED_SLOT: _normalize_slot,
    FactType.ORIENTATION_SLOT: _normalize_slot,
    FactType.ORIENTATION_READY: _normalize_bool,
    FactType.FAQ_ANSWER: lambda v: "" if v is None else str(v).strip(),
}


def normalize_fact_value(fact_type: FactType, value: Any) -> Any:
    """Normalize raw fact values according to their type."""
    normalizer = _FACT_NORMALIZERS.get(fact_type)
    return normalizer(value) if normalizer else value


StateRule = Dict[str, Any]


STATE_REQUIREMENTS: Dict[str, StateRule] = {
    "WELCOME": {
        "required": [FactType.CONSENT],
        "next_state": "ELIGIBILITY_PART1",
        "prompts": {
            FactType.CONSENT: "Are you ready to start the onboarding process?",
        },
    },
    "ELIGIBILITY_PART1": {
        "required": [FactType.AGE, FactType.DEVICE],
        "next_state": "ELIGIBILITY_PART2",
        "prompts": {
            FactType.AGE: "Could you confirm your age (18+)?",
            FactType.DEVICE: "Do you have a smartphone or laptop with reliable internet?",
        },
    },
    "ELIGIBILITY_PART2": {
        "required": [FactType.COMMITMENT_HOURS],
        "optional": [FactType.COMMITMENT_MONTHS, FactType.PREFERRED_DAY, FactType.PREFERRED_TIME],
        "next_state": "PREFS_DAYTIME",
        "prompts": {
            FactType.COMMITMENT_HOURS: "How many hours per week can you volunteer (around 2)?",
        },
    },
    "PREFS_DAYTIME": {
        "required": [FactType.PREFERRED_DAY, FactType.PREFERRED_TIME],
        "next_state": "QA_WINDOW",
        "prompts": {
            FactType.PREFERRED_DAY: "Which weekdays between Monday and Friday can you teach?",
            FactType.PREFERRED_TIME: "What time of day works best between 8 AM and 3 PM?",
        },
    },
    "QA_WINDOW": {
        "required": [FactType.FAQ_ANSWER],
        "optional": [FactType.ORIENTATION_READY],
        "next_state": "ORIENTATION",
        "prompts": {
            FactType.FAQ_ANSWER: "Let me know what else you’d like to clarify.",
        },
    },
    "ORIENTATION": {
        "required": [FactType.ORIENTATION_SLOT],
        "next_state": "WRAP",
        "prompts": {
            FactType.ORIENTATION_SLOT: "Which orientation slot works for you?",
        },
    },
}


def evaluate_state_progress(
    current_state: str,
    facts: Sequence[Fact],
    *,
    min_confidence: float = 0.7,
) -> StateEvaluation:
    """
    Evaluate whether the collected facts satisfy the requirements of the current state.

    Returns:
        StateEvaluation with recommended next_state, satisfied fact types,
        and prompts for any missing information.
    """

    state_key = current_state.upper()
    rule = STATE_REQUIREMENTS.get(state_key)
    if not rule:
        return StateEvaluation(next_state="SAME")

    required: Iterable[FactType] = rule.get("required", [])
    satisfied: List[FactType] = []
    missing: List[FactType] = []
    pending_questions: List[str] = []

    def _has_fact(fact_type: FactType) -> bool:
        for fact in facts:
            if fact.type == fact_type and (fact.confidence or 0.0) >= min_confidence:
                return True
        return False

    for fact_type in required:
        if _has_fact(fact_type):
            satisfied.append(fact_type)
        else:
            missing.append(fact_type)
            prompt = rule.get("prompts", {}).get(fact_type)
            if not pending_questions:
                if prompt:
                    pending_questions.append(prompt)
                else:
                    pending_questions.append(f"Could you share your {fact_type.value.replace('_', ' ')}?")

    next_state = rule.get("next_state", "SAME") if not missing else "SAME"

    return StateEvaluation(
        next_state=next_state,
        satisfied=satisfied,
        missing=missing,
        pending_questions=pending_questions,
    )

