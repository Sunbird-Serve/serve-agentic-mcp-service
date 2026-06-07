"""
State MCP: Conversation State Store
- Save and retrieve conversation state so agents can resume flows
- File-based persistence (JSONL) for durability across restarts
- Configurable TTL/expiry for stale states
- Supports full conversation state (not just onboarding step)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone, timedelta
import json
import os
import uuid
import threading

router = APIRouter()

# --------- Configuration ---------

STATE_STORE_DIR = os.environ.get("STATE_STORE_DIR", "./state_store")
STATE_TTL_HOURS = int(os.environ.get("STATE_TTL_HOURS", "72"))  # 3 days default

# --------- State Machine (Onboarding) ---------

VALID_STATES = {
    "WELCOME", "ELIGIBILITY_PART1", "ELIGIBILITY_PART2", "TIME_PREF",
    "SCHEDULING", "DONE", "REJECTED", "DEFERRED", "OPTOUT",
    "QA_WINDOW", "ORIENTATION_CONSENT", "ORIENTATION_SCHEDULING"
}

FORWARD_TRANSITIONS = {
    "WELCOME": {"to_ELIGIBILITY_PART1"},
    "ELIGIBILITY_PART1": {"to_ELIGIBILITY_PART2"},
    "ELIGIBILITY_PART2": {"to_TIME_PREF"},
    "TIME_PREF": {"to_SCHEDULING"},
    "SCHEDULING": {"to_DONE"}
}

TERMINAL_STATES = {"REJECTED", "DEFERRED", "OPTOUT"}

INTENT_TO_STATE = {
    "to_ELIGIBILITY_PART1": "ELIGIBILITY_PART1",
    "to_ELIGIBILITY_PART2": "ELIGIBILITY_PART2",
    "to_TIME_PREF": "TIME_PREF",
    "to_SCHEDULING": "SCHEDULING",
    "to_DONE": "DONE",
    "to_REJECTED": "REJECTED",
    "to_DEFERRED": "DEFERRED",
    "to_OPTOUT": "OPTOUT",
    "ORIENTATION_CONSENT": "ORIENTATION_CONSENT",
    "to_ORIENTATION_CONSENT": "ORIENTATION_CONSENT",
    "to_ORIENTATION_SCHEDULING": "ORIENTATION_SCHEDULING"
}

STATE_REQUIRED_FIELDS = {
    "ELIGIBILITY_PART1": ["age_ok", "has_device"],
    "ELIGIBILITY_PART2": ["weekly_commitment_hours"],
    "TIME_PREF": ["time_band_preference"],
    "SCHEDULING": ["slot_selection"],
    "QA_WINDOW": [],
    "ORIENTATION_CONSENT": ["consent_given"],
    "ORIENTATION_SCHEDULING": ["slot_selection"],
    "DONE": []
}

# --------- Persistent Store ---------

_STATE_STORE: Dict[str, Dict] = {}
_TRANSITION_STORE: Dict[str, Dict] = {}
_CONVERSATION_STORE: Dict[str, Dict] = {}
_lock = threading.Lock()
_loaded = False


def _ensure_store_dir():
    os.makedirs(STATE_STORE_DIR, exist_ok=True)


def _state_file_path() -> str:
    return os.path.join(STATE_STORE_DIR, "states.jsonl")


def _conversation_file_path() -> str:
    return os.path.join(STATE_STORE_DIR, "conversations.jsonl")


def _load_from_disk():
    """Load state from JSONL files on first access."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    _ensure_store_dir()

    # Load onboarding states
    state_path = _state_file_path()
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        vid = record.get("volunteerId")
                        if vid:
                            _STATE_STORE[vid] = record
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[state] Warning: Failed to load states: {e}")

    # Load conversation states
    conv_path = _conversation_file_path()
    if os.path.exists(conv_path):
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        vid = record.get("volunteerId")
                        if vid:
                            _CONVERSATION_STORE[vid] = record
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[state] Warning: Failed to load conversations: {e}")


def _persist_state(volunteer_id: str, record: Dict):
    """Append state record to JSONL file."""
    try:
        _ensure_store_dir()
        record["volunteerId"] = volunteer_id
        with open(_state_file_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[state] Warning: Failed to persist state: {e}")


def _persist_conversation(volunteer_id: str, record: Dict):
    """Append conversation state to JSONL file."""
    try:
        _ensure_store_dir()
        record["volunteerId"] = volunteer_id
        with open(_conversation_file_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[state] Warning: Failed to persist conversation: {e}")


def _is_expired(record: Dict) -> bool:
    """Check if a state record has expired based on TTL."""
    updated_at = record.get("updated_at")
    if not updated_at:
        return False
    try:
        ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        expiry = ts + timedelta(hours=STATE_TTL_HOURS)
        return datetime.now(timezone.utc) > expiry
    except (ValueError, TypeError):
        return False


# --------- Models ---------


class StateGetRequest(BaseModel):
    volunteerId: str = Field(..., description="Unique volunteer identifier")


class StateMetadata(BaseModel):
    last_completed_step: Optional[str] = None
    progress_percentage: Optional[int] = Field(None, ge=0, le=100)
    flags: Optional[Dict[str, Any]] = None


class StateGetResponse(BaseModel):
    state: str
    updated_at: Optional[str] = None
    metadata: Optional[StateMetadata] = None
    expired: bool = False


class StateAdvanceRequest(BaseModel):
    volunteerId: str = Field(..., description="Unique volunteer identifier")
    intent: str = Field(..., description="Intent to advance (e.g., 'to_ELIGIBILITY_PART1')")
    idempotency_key: Optional[str] = Field(None, description="Optional idempotency key")


class StateAdvanceResponse(BaseModel):
    new_state: str
    previous_state: str
    transitioned_at: str
    next_required_fields: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationSaveRequest(BaseModel):
    volunteerId: str = Field(..., description="Volunteer ID")
    state: Dict[str, Any] = Field(..., description="Full conversation state object to persist")
    ttl_hours: Optional[int] = Field(None, ge=1, le=720, description="Custom TTL in hours (default: 72)")


class ConversationSaveResponse(BaseModel):
    saved: bool
    volunteerId: str
    expires_at: str
    version: int


class ConversationGetRequest(BaseModel):
    volunteerId: str = Field(..., description="Volunteer ID")


class ConversationGetResponse(BaseModel):
    found: bool
    volunteerId: str
    state: Optional[Dict[str, Any]] = None
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None
    expired: bool = False
    version: int = 0


# --------- Helper Functions ---------


def _calculate_progress(state: str) -> int:
    progress_map = {
        "WELCOME": 0, "ELIGIBILITY_PART1": 25, "ELIGIBILITY_PART2": 40,
        "TIME_PREF": 60, "SCHEDULING": 80, "QA_WINDOW": 70,
        "ORIENTATION_CONSENT": 75, "ORIENTATION_SCHEDULING": 85,
        "DONE": 100, "REJECTED": 0, "DEFERRED": 0, "OPTOUT": 0
    }
    return progress_map.get(state, 0)


def _get_last_completed_step(state: str) -> Optional[str]:
    step_map = {
        "ELIGIBILITY_PART1": "WELCOME", "ELIGIBILITY_PART2": "ELIGIBILITY_PART1",
        "TIME_PREF": "ELIGIBILITY_PART2", "SCHEDULING": "TIME_PREF", "DONE": "SCHEDULING"
    }
    return step_map.get(state)


def _validate_transition(current_state: str, intent: str, target_state: str) -> bool:
    if target_state in TERMINAL_STATES:
        return True
    if current_state in FORWARD_TRANSITIONS:
        if intent in FORWARD_TRANSITIONS[current_state]:
            return True
    # Allow same-state (idempotent)
    if current_state == target_state:
        return True
    # Allow orientation states from any non-terminal state
    if target_state in ("ORIENTATION_CONSENT", "ORIENTATION_SCHEDULING"):
        return True
    return False


# --------- Endpoints ---------


@router.post("/state.get", response_model=StateGetResponse)
async def get_state(req: StateGetRequest) -> StateGetResponse:
    """Get the current onboarding state for a volunteer."""
    with _lock:
        _load_from_disk()

    volunteer_id = req.volunteerId

    if volunteer_id in _STATE_STORE:
        state_record = _STATE_STORE[volunteer_id]
        current_state = state_record.get("state", "WELCOME")
        updated_at = state_record.get("updated_at")
        expired = _is_expired(state_record)

        metadata = StateMetadata(
            last_completed_step=_get_last_completed_step(current_state),
            progress_percentage=_calculate_progress(current_state),
            flags=state_record.get("flags", {})
        )

        return StateGetResponse(
            state=current_state,
            updated_at=updated_at,
            metadata=metadata,
            expired=expired,
        )

    return StateGetResponse(
        state="WELCOME",
        updated_at=None,
        metadata=StateMetadata(last_completed_step=None, progress_percentage=0, flags={}),
        expired=False,
    )


@router.post("/state.advance", response_model=StateAdvanceResponse)
async def advance_state(req: StateAdvanceRequest) -> StateAdvanceResponse:
    """Advance a volunteer's onboarding state based on intent."""
    with _lock:
        _load_from_disk()

    volunteer_id = req.volunteerId
    intent = req.intent

    if intent not in INTENT_TO_STATE:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid intent: '{intent}'. Valid intents: {list(INTENT_TO_STATE.keys())}"
        )

    target_state = INTENT_TO_STATE[intent]

    current_state = "WELCOME"
    if volunteer_id in _STATE_STORE:
        current_state = _STATE_STORE[volunteer_id].get("state", "WELCOME")

    if target_state not in VALID_STATES:
        raise HTTPException(status_code=422, detail=f"Invalid target state: '{target_state}'")

    idempotency_key = req.idempotency_key or f"{volunteer_id}_{intent}_{int(datetime.now(timezone.utc).timestamp())}"

    if idempotency_key in _TRANSITION_STORE:
        existing = _TRANSITION_STORE[idempotency_key]
        return StateAdvanceResponse(
            new_state=existing["new_state"],
            previous_state=existing["previous_state"],
            transitioned_at=existing["transitioned_at"],
            next_required_fields=existing.get("next_required_fields"),
            metadata=existing.get("metadata", {})
        )

    if not _validate_transition(current_state, intent, target_state):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid transition: cannot advance from '{current_state}' to '{target_state}' with intent '{intent}'"
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    previous_state = current_state

    with _lock:
        if volunteer_id not in _STATE_STORE:
            _STATE_STORE[volunteer_id] = {}
        _STATE_STORE[volunteer_id]["state"] = target_state
        _STATE_STORE[volunteer_id]["updated_at"] = now_iso
        if "flags" not in _STATE_STORE[volunteer_id]:
            _STATE_STORE[volunteer_id]["flags"] = {}

        transition_id = f"trans_{uuid.uuid4().hex[:12]}"
        transition_record = {
            "new_state": target_state,
            "previous_state": previous_state,
            "transitioned_at": now_iso,
            "next_required_fields": STATE_REQUIRED_FIELDS.get(target_state),
            "metadata": {
                "transition_id": transition_id,
                "triggered_by": intent,
                "volunteer_id": volunteer_id
            }
        }
        _TRANSITION_STORE[idempotency_key] = transition_record

    _persist_state(volunteer_id, _STATE_STORE[volunteer_id])

    print(f"[state.advance] {volunteer_id}: {previous_state} -> {target_state} (intent: {intent})")

    return StateAdvanceResponse(
        new_state=target_state,
        previous_state=previous_state,
        transitioned_at=now_iso,
        next_required_fields=STATE_REQUIRED_FIELDS.get(target_state),
        metadata=transition_record["metadata"]
    )


@router.post("/conversation.save", response_model=ConversationSaveResponse)
async def conversation_save(req: ConversationSaveRequest) -> ConversationSaveResponse:
    """
    Save full conversation state for a volunteer.

    This stores arbitrary JSON state (context, collected facts, partial responses, etc.)
    so that agents can resume flows after disconnection or handoff.
    """
    with _lock:
        _load_from_disk()

    now = datetime.now(timezone.utc)
    ttl = req.ttl_hours or STATE_TTL_HOURS
    expires_at = now + timedelta(hours=ttl)

    version = 1
    if req.volunteerId in _CONVERSATION_STORE:
        version = _CONVERSATION_STORE[req.volunteerId].get("version", 0) + 1

    record = {
        "state": req.state,
        "updated_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "version": version,
        "ttl_hours": ttl,
    }

    with _lock:
        _CONVERSATION_STORE[req.volunteerId] = record

    _persist_conversation(req.volunteerId, record)

    return ConversationSaveResponse(
        saved=True,
        volunteerId=req.volunteerId,
        expires_at=expires_at.isoformat(),
        version=version,
    )


@router.post("/conversation.get", response_model=ConversationGetResponse)
async def conversation_get(req: ConversationGetRequest) -> ConversationGetResponse:
    """
    Retrieve conversation state for a volunteer.

    Returns the stored state along with expiry info. If expired, the state is still
    returned (for debugging) but marked as expired.
    """
    with _lock:
        _load_from_disk()

    if req.volunteerId not in _CONVERSATION_STORE:
        return ConversationGetResponse(
            found=False,
            volunteerId=req.volunteerId,
        )

    record = _CONVERSATION_STORE[req.volunteerId]
    expires_at = record.get("expires_at")

    expired = False
    if expires_at:
        try:
            exp_dt = datetime.fromisoformat(expires_at)
            expired = datetime.now(timezone.utc) > exp_dt
        except (ValueError, TypeError):
            pass

    return ConversationGetResponse(
        found=True,
        volunteerId=req.volunteerId,
        state=record.get("state"),
        updated_at=record.get("updated_at"),
        expires_at=expires_at,
        expired=expired,
        version=record.get("version", 0),
    )
