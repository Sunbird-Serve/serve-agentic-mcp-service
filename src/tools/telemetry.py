"""
Telemetry MCP Tool — Logging SIA Events
- Log agent events (drop-offs, failures, completions, state transitions)
- Persist events to JSONL file for analytics
- Query events by volunteerId, eventType, or time range
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import json
import os
import threading

router = APIRouter()

# --------- Configuration ---------

TELEMETRY_LOG_DIR = os.environ.get("TELEMETRY_LOG_DIR", "./telemetry_logs")
TELEMETRY_MAX_MEMORY_EVENTS = int(os.environ.get("TELEMETRY_MAX_MEMORY_EVENTS", "1000"))

# --------- Event Schema ---------

VALID_EVENT_TYPES = {
    "onboarding.started",
    "onboarding.step_completed",
    "onboarding.completed",
    "onboarding.dropped_off",
    "onboarding.deferred",
    "onboarding.rejected",
    "eligibility.passed",
    "eligibility.failed",
    "scheduling.slot_proposed",
    "scheduling.slot_confirmed",
    "scheduling.slot_declined",
    "volunteer.registered",
    "volunteer.nominated",
    "volunteer.status_updated",
    "tool.called",
    "tool.failed",
    "tool.timeout",
    "message.sent",
    "message.received",
    "error.unhandled",
    "state.advanced",
    "faq.answered",
    "video.sent",
}

# --------- Models ---------


class TelemetryEvent(BaseModel):
    event: str = Field(..., description="Event type (e.g., 'onboarding.completed')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload data")
    volunteerId: Optional[str] = Field(None, description="Volunteer ID if applicable")
    sessionId: Optional[str] = Field(None, description="Session/conversation ID")
    timestamp: Optional[str] = Field(None, description="ISO8601 timestamp (auto-generated if not provided)")


class TelemetryEmitResponse(BaseModel):
    ok: bool
    eventId: str
    timestamp: str
    event: str


class TelemetryQueryRequest(BaseModel):
    volunteerId: Optional[str] = Field(None, description="Filter by volunteer ID")
    eventType: Optional[str] = Field(None, description="Filter by event type (prefix match supported)")
    since: Optional[str] = Field(None, description="ISO8601 timestamp — only events after this time")
    until: Optional[str] = Field(None, description="ISO8601 timestamp — only events before this time")
    limit: int = Field(default=50, ge=1, le=200, description="Max events to return")


class TelemetryQueryResponse(BaseModel):
    events: List[Dict[str, Any]]
    total: int
    truncated: bool


class TelemetryStatsRequest(BaseModel):
    volunteerId: Optional[str] = Field(None, description="Filter stats by volunteer ID")
    since: Optional[str] = Field(None, description="ISO8601 timestamp — stats after this time")


class TelemetryStatsResponse(BaseModel):
    total_events: int
    by_type: Dict[str, int]
    unique_volunteers: int
    time_range: Dict[str, Optional[str]]


# --------- In-Memory Store + File Persistence ---------

_events: List[Dict[str, Any]] = []
_lock = threading.Lock()
_event_counter = 0


def _ensure_log_dir():
    """Create telemetry log directory if it doesn't exist."""
    os.makedirs(TELEMETRY_LOG_DIR, exist_ok=True)


def _get_log_file_path() -> str:
    """Get current log file path (one file per day)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(TELEMETRY_LOG_DIR, f"events_{today}.jsonl")


def _persist_event(event_record: Dict[str, Any]):
    """Append event to JSONL log file."""
    try:
        _ensure_log_dir()
        log_path = _get_log_file_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[telemetry] Warning: Failed to persist event: {e}")


def _generate_event_id() -> str:
    """Generate a unique event ID."""
    global _event_counter
    _event_counter += 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"evt_{ts}_{_event_counter:06d}"


def _load_events_from_disk(since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Load events from JSONL files on disk (for queries spanning beyond memory)."""
    events = []
    try:
        _ensure_log_dir()
        files = sorted(os.listdir(TELEMETRY_LOG_DIR))
        for fname in files:
            if not fname.startswith("events_") or not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(TELEMETRY_LOG_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if since:
                            evt_time = datetime.fromisoformat(record.get("timestamp", "").replace("Z", "+00:00").replace("+00:00", ""))
                            if evt_time < since:
                                continue
                        events.append(record)
                    except (json.JSONDecodeError, ValueError):
                        continue
    except Exception as e:
        print(f"[telemetry] Warning: Failed to load events from disk: {e}")
    return events


# --------- Endpoints ---------


@router.post("/telemetry.emit", response_model=TelemetryEmitResponse)
async def telemetry_emit(req: TelemetryEvent) -> TelemetryEmitResponse:
    """
    Log a telemetry event.

    Accepts any event type (known types are validated but unknown types are still logged
    with a warning flag). Events are stored in memory and persisted to JSONL files.
    """
    now = datetime.now(timezone.utc)
    timestamp = req.timestamp or (now.isoformat() + "Z")
    event_id = _generate_event_id()

    event_record = {
        "eventId": event_id,
        "event": req.event,
        "timestamp": timestamp,
        "payload": req.payload,
        "volunteerId": req.volunteerId,
        "sessionId": req.sessionId,
    }

    if req.event not in VALID_EVENT_TYPES:
        event_record["_warning"] = f"Unknown event type: {req.event}"

    with _lock:
        _events.append(event_record)
        if len(_events) > TELEMETRY_MAX_MEMORY_EVENTS:
            _events.pop(0)

    _persist_event(event_record)

    print(f"[telemetry] {req.event} | vol={req.volunteerId or '-'} | session={req.sessionId or '-'}")

    return TelemetryEmitResponse(
        ok=True,
        eventId=event_id,
        timestamp=timestamp,
        event=req.event,
    )


@router.post("/telemetry.query", response_model=TelemetryQueryResponse)
async def telemetry_query(req: TelemetryQueryRequest) -> TelemetryQueryResponse:
    """
    Query telemetry events with filtering.

    Searches in-memory events first, falls back to disk if needed.
    """
    with _lock:
        source_events = list(_events)

    if req.since:
        try:
            since_dt = datetime.fromisoformat(req.since.replace("Z", ""))
            disk_events = _load_events_from_disk(since=since_dt)
            seen_ids = {e["eventId"] for e in source_events}
            for de in disk_events:
                if de.get("eventId") not in seen_ids:
                    source_events.append(de)
        except ValueError:
            pass

    filtered = []
    for evt in source_events:
        if req.volunteerId and evt.get("volunteerId") != req.volunteerId:
            continue
        if req.eventType:
            if not evt.get("event", "").startswith(req.eventType):
                continue
        if req.since:
            evt_ts = evt.get("timestamp", "")
            if evt_ts < req.since:
                continue
        if req.until:
            evt_ts = evt.get("timestamp", "")
            if evt_ts > req.until:
                continue
        filtered.append(evt)

    filtered.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    total = len(filtered)
    truncated = total > req.limit
    results = filtered[: req.limit]

    return TelemetryQueryResponse(
        events=results,
        total=total,
        truncated=truncated,
    )


@router.post("/telemetry.stats", response_model=TelemetryStatsResponse)
async def telemetry_stats(req: TelemetryStatsRequest) -> TelemetryStatsResponse:
    """
    Get aggregate telemetry statistics.

    Returns event counts by type, unique volunteers, and time range.
    """
    with _lock:
        source_events = list(_events)

    if req.since:
        try:
            since_dt = datetime.fromisoformat(req.since.replace("Z", ""))
            disk_events = _load_events_from_disk(since=since_dt)
            seen_ids = {e["eventId"] for e in source_events}
            for de in disk_events:
                if de.get("eventId") not in seen_ids:
                    source_events.append(de)
        except ValueError:
            pass

    filtered = source_events
    if req.volunteerId:
        filtered = [e for e in filtered if e.get("volunteerId") == req.volunteerId]
    if req.since:
        filtered = [e for e in filtered if e.get("timestamp", "") >= req.since]

    by_type: Dict[str, int] = {}
    volunteers = set()
    timestamps = []

    for evt in filtered:
        etype = evt.get("event", "unknown")
        by_type[etype] = by_type.get(etype, 0) + 1
        vid = evt.get("volunteerId")
        if vid:
            volunteers.add(vid)
        ts = evt.get("timestamp")
        if ts:
            timestamps.append(ts)

    time_range: Dict[str, Optional[str]] = {"earliest": None, "latest": None}
    if timestamps:
        timestamps.sort()
        time_range["earliest"] = timestamps[0]
        time_range["latest"] = timestamps[-1]

    return TelemetryStatsResponse(
        total_events=len(filtered),
        by_type=by_type,
        unique_volunteers=len(volunteers),
        time_range=time_range,
    )
