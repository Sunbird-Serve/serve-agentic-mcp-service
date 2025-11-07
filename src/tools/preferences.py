from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from tools.telemetry import TelemetryEvent, telemetry_emit
import uuid

router = APIRouter()

_PREFS: Dict[str, Dict] = {}

class TimeWindow(BaseModel):
    start: str
    end: str

class PrefsBody(BaseModel):
    days: List[str] = Field(description="Preferred weekdays, e.g., ['Mon','Tue']")
    # Accept either strings "HH:MM-HH:MM" or objects {start,end}; normalize to objects
    time_windows: List[TimeWindow] = Field(description="Preferred time windows, e.g., ['08:00-11:00'] or [{start,end}]")
    timezone: str = Field(description="IANA timezone, e.g., 'Asia/Kolkata'")
    
    @field_validator('days')
    @classmethod
    def validate_days(cls, v: List[str]):
        # Accept ISO uppercase codes and title-case weekdays
        valid_upper = {"MON","TUE","WED","THU","FRI","SAT","SUN"}
        if not v:
            raise ValueError("days must not be empty")
        for d in v:
            d_norm = (d or "").strip()
            if d_norm.upper() not in valid_upper and d_norm.title() not in {"Mon","Tue","Wed","Thu","Fri","Sat","Sun"}:
                raise ValueError(f"invalid day: {d}")
        return v
    
    @field_validator('time_windows', mode='before')
    @classmethod
    def validate_windows(cls, v: Any):
        if not v:
            raise ValueError("time_windows must not be empty")
        normalized: List[TimeWindow] = []
        for w in v:
            if isinstance(w, dict):
                start = w.get('start')
                end = w.get('end')
                if not start or not end:
                    raise ValueError(f"invalid time window object: {w}")
                normalized.append(TimeWindow(start=str(start), end=str(end)))
            elif isinstance(w, str):
                if '-' not in w or len(w) < 11:
                    raise ValueError(f"invalid time window: {w}")
                start, end = w.split('-', 1)
                start = start.strip()
                end = end.strip()
                if not start or not end:
                    raise ValueError(f"invalid time window: {w}")
                normalized.append(TimeWindow(start=start, end=end))
            else:
                raise ValueError(f"invalid time window type: {type(w).__name__}")
        return normalized

class PrefsSaveRequest(BaseModel):
    volunteerId: str
    prefs: PrefsBody
    policy_version: Optional[str] = None
    idempotency_key: Optional[str] = None

class PrefsSaveResponse(BaseModel):
    saved: bool
    prefs_id: Optional[str] = None
    policy_version: Optional[str] = None

_IDEMPOTENCY: Dict[str, str] = {}

@router.post("/preferences.save", response_model=PrefsSaveResponse)
async def preferences_save(req: PrefsSaveRequest) -> PrefsSaveResponse:
    # Basic validation beyond pydantic (e.g., enforce weekday-only policy)
    for d in req.prefs.days:
        d_up = d.upper()
        if d_up not in {"MON","TUE","WED","THU","FRI"}:
            raise HTTPException(status_code=422, detail={"field": "prefs.days", "message": f"Weekend not allowed: {d}"})
    # Idempotency: return existing prefs_id if key seen
    if req.idempotency_key and req.idempotency_key in _IDEMPOTENCY:
        pid = _IDEMPOTENCY[req.idempotency_key]
        return PrefsSaveResponse(saved=True, prefs_id=pid, policy_version=req.policy_version)

    # Save
    prefs_id = f"prefs_{uuid.uuid4().hex[:12]}"
    _PREFS[req.volunteerId] = {
        "prefs_id": prefs_id,
        "prefs": req.prefs.model_dump(),
        "policy_version": req.policy_version,
        "idempotency_key": req.idempotency_key,
    }
    if req.idempotency_key:
        _IDEMPOTENCY[req.idempotency_key] = prefs_id
    # Emit telemetry
    try:
        await telemetry_emit(TelemetryEvent(event="onboarding.prefs_saved", payload={
            "volunteerId": req.volunteerId,
            "policy_version": req.policy_version,
            "prefs": req.prefs.model_dump()
        }))
    except Exception:
        pass
    return PrefsSaveResponse(saved=True, prefs_id=prefs_id, policy_version=req.policy_version)
