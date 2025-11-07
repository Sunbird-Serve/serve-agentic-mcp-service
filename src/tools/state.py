"""
State Management Tool
- Get current onboarding state for a volunteer
- Advance state based on intent (with validation)
- Supports idempotency for state transitions
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
from datetime import datetime
import uuid

router = APIRouter()

# In-memory storage (temporary - will be replaced with database)
_STATE_STORE: Dict[str, Dict] = {}  # Key: volunteerId
_TRANSITION_STORE: Dict[str, Dict] = {}  # Key: idempotency_key

# State machine: valid transitions
VALID_STATES = {
    "WELCOME", "ELIGIBILITY_PART1", "ELIGIBILITY_PART2", "TIME_PREF",
    "SCHEDULING", "DONE", "REJECTED", "DEFERRED", "OPTOUT",
    "QA_WINDOW", "ORIENTATION_CONSENT", "ORIENTATION_SCHEDULING"
}

# Forward-only normal flow transitions
FORWARD_TRANSITIONS = {
    "WELCOME": {"to_ELIGIBILITY_PART1"},
    "ELIGIBILITY_PART1": {"to_ELIGIBILITY_PART2"},
    "ELIGIBILITY_PART2": {"to_TIME_PREF"},
    "TIME_PREF": {"to_SCHEDULING"},
    "SCHEDULING": {"to_DONE"}
}

# Terminal states (can be reached from any state)
TERMINAL_STATES = {"REJECTED", "DEFERRED", "OPTOUT"}

# Intent to state mapping
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

# Required fields for each state (for next_required_fields in response)
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

class StateGetRequest(BaseModel):
    volunteerId: str = Field(..., description="Unique volunteer identifier")

class StateMetadata(BaseModel):
    last_completed_step: Optional[str] = None
    progress_percentage: Optional[int] = Field(None, ge=0, le=100)
    flags: Optional[Dict[str, Any]] = None

class StateGetResponse(BaseModel):
    state: str
    updated_at: Optional[str] = None  # ISO8601 datetime or null
    metadata: Optional[StateMetadata] = None

class StateAdvanceRequest(BaseModel):
    volunteerId: str = Field(..., description="Unique volunteer identifier")
    intent: str = Field(..., description="Intent to advance (e.g., 'to_ELIGIBILITY_PART1')")
    idempotency_key: Optional[str] = Field(None, description="Optional idempotency key to prevent duplicate transitions")

class StateAdvanceResponse(BaseModel):
    new_state: str
    previous_state: str
    transitioned_at: str  # ISO8601 datetime
    next_required_fields: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

def _calculate_progress(state: str) -> int:
    """Calculate progress percentage based on state"""
    progress_map = {
        "WELCOME": 0,
        "ELIGIBILITY_PART1": 25,
        "ELIGIBILITY_PART2": 40,
        "TIME_PREF": 60,
        "SCHEDULING": 80,
        "QA_WINDOW": 70,
        "ORIENTATION_CONSENT": 75,
        "ORIENTATION_SCHEDULING": 85,
        "DONE": 100,
        "REJECTED": 0,
        "DEFERRED": 0,
        "OPTOUT": 0
    }
    return progress_map.get(state, 0)

def _get_last_completed_step(state: str) -> Optional[str]:
    """Get last completed step based on current state"""
    step_map = {
        "ELIGIBILITY_PART1": "WELCOME",
        "ELIGIBILITY_PART2": "ELIGIBILITY_PART1",
        "TIME_PREF": "ELIGIBILITY_PART2",
        "SCHEDULING": "TIME_PREF",
        "DONE": "SCHEDULING"
    }
    return step_map.get(state)

def _validate_transition(current_state: str, intent: str, target_state: str) -> bool:
    """Validate if transition from current_state to target_state is valid"""
    # Terminal states can be reached from any state
    if target_state in TERMINAL_STATES:
        return True
    
    # Check forward transitions
    if current_state in FORWARD_TRANSITIONS:
        allowed_intents = FORWARD_TRANSITIONS[current_state]
        if intent in allowed_intents:
            return True
    
    # Check if intent maps to target state correctly
    if INTENT_TO_STATE.get(intent) == target_state:
        # Additional validation: prevent backwards transitions
        state_order = ["WELCOME", "ELIGIBILITY_PART1", "ELIGIBILITY_PART2", "TIME_PREF", "SCHEDULING", "DONE"]
        try:
            current_idx = state_order.index(current_state)
            target_idx = state_order.index(target_state)
            if target_idx > current_idx:  # Forward transition
                return True
            elif target_idx == current_idx:  # Same state (no-op, but allow)
                return True
            else:  # Backwards transition
                return False
        except ValueError:
            # States not in order list (terminal states), allow if intent matches
            return True
    
    return False

@router.post("/state.get", response_model=StateGetResponse)
async def get_state(req: StateGetRequest) -> StateGetResponse:
    """
    Get the current onboarding state for a volunteer.
    
    Returns default "WELCOME" state if volunteer has no state record.
    """
    volunteer_id = req.volunteerId
    
    if volunteer_id in _STATE_STORE:
        state_record = _STATE_STORE[volunteer_id]
        current_state = state_record.get("state", "WELCOME")
        updated_at = state_record.get("updated_at")
        
        metadata = StateMetadata(
            last_completed_step=_get_last_completed_step(current_state),
            progress_percentage=_calculate_progress(current_state),
            flags=state_record.get("flags", {})
        )
        
        return StateGetResponse(
            state=current_state,
            updated_at=updated_at,
            metadata=metadata
        )
    
    # New volunteer - return default state
    return StateGetResponse(
        state="WELCOME",
        updated_at=None,
        metadata=StateMetadata(
            last_completed_step=None,
            progress_percentage=0,
            flags={}
        )
    )

@router.post("/state.advance", response_model=StateAdvanceResponse)
async def advance_state(req: StateAdvanceRequest) -> StateAdvanceResponse:
    """
    Advance a volunteer's onboarding state based on intent.
    
    Validates transition is allowed (prevents invalid backwards/invalid moves).
    Supports idempotency to prevent duplicate transitions.
    """
    volunteer_id = req.volunteerId
    intent = req.intent
    
    # Validate intent exists
    if intent not in INTENT_TO_STATE:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid intent: '{intent}'. Valid intents: {list(INTENT_TO_STATE.keys())}"
        )
    
    target_state = INTENT_TO_STATE[intent]
    
    # Get current state
    current_state = "WELCOME"
    if volunteer_id in _STATE_STORE:
        current_state = _STATE_STORE[volunteer_id].get("state", "WELCOME")
    
    # Validate target state
    if target_state not in VALID_STATES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid target state: '{target_state}'. Valid states: {list(VALID_STATES)}"
        )
    
    # Generate idempotency key if not provided
    idempotency_key = req.idempotency_key
    if not idempotency_key:
        idempotency_key = f"{volunteer_id}_{intent}_{int(datetime.utcnow().timestamp())}"
    
    # Check for existing transition (idempotency)
    if idempotency_key in _TRANSITION_STORE:
        existing = _TRANSITION_STORE[idempotency_key]
        print(f"[state.advance] Found existing transition for idempotency_key: {idempotency_key}")
        return StateAdvanceResponse(
            new_state=existing["new_state"],
            previous_state=existing["previous_state"],
            transitioned_at=existing["transitioned_at"],
            next_required_fields=existing.get("next_required_fields"),
            metadata=existing.get("metadata", {})
        )
    
    # Validate transition
    if not _validate_transition(current_state, intent, target_state):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid transition: cannot advance from '{current_state}' to '{target_state}' with intent '{intent}'"
        )
    
    # Perform transition
    now_iso = datetime.utcnow().isoformat() + "Z"
    previous_state = current_state
    
    # Update state store
    if volunteer_id not in _STATE_STORE:
        _STATE_STORE[volunteer_id] = {}
    
    _STATE_STORE[volunteer_id]["state"] = target_state
    _STATE_STORE[volunteer_id]["updated_at"] = now_iso
    if "flags" not in _STATE_STORE[volunteer_id]:
        _STATE_STORE[volunteer_id]["flags"] = {}
    
    # Store transition record
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
    
    print(f"[state.advance] Transitioned {volunteer_id}: {previous_state} → {target_state} (intent: {intent})")
    
    return StateAdvanceResponse(
        new_state=target_state,
        previous_state=previous_state,
        transitioned_at=now_iso,
        next_required_fields=STATE_REQUIRED_FIELDS.get(target_state),
        metadata=transition_record["metadata"]
    )

