"""
Deferral Management Tool
- Create deferral records when volunteers want to postpone onboarding
- Supports idempotency to prevent duplicate deferrals
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional
from datetime import datetime, timezone
import uuid
import re

router = APIRouter()

# In-memory storage (temporary - will be replaced with database)
_DEFERRAL_STORE: Dict[str, Dict] = {}  # Key: idempotency_key (if provided) or volunteerId
_DEFERRAL_BY_VOLUNTEER: Dict[str, str] = {}  # volunteerId -> idempotency_key mapping

class DeferralCreateRequest(BaseModel):
    volunteerId: str = Field(..., description="Unique volunteer identifier")
    reason: str = Field(..., description="Reason for deferral (e.g., 'user_requested_later', 'ORIENTATION_LATER')")
    until_ISO: str = Field(..., description="ISO8601 datetime when to contact again")
    idempotency_key: Optional[str] = Field(None, description="Optional idempotency key to prevent duplicates")
    
    @field_validator('volunteerId')
    @classmethod
    def validate_volunteer_id(cls, v):
        if not v or not v.strip():
            raise ValueError("volunteerId is required and cannot be empty")
        return v.strip()
    
    @field_validator('until_ISO')
    @classmethod
    def validate_until_iso(cls, v):
        """Validate ISO8601 datetime format"""
        if not v or not v.strip():
            raise ValueError("until_ISO is required and cannot be empty")
        
        # More flexible ISO8601 validation - allow microseconds (0-6 digits) and various timezone formats
        v_stripped = v.strip()
        
        # Check if datetime is parseable (Python's fromisoformat is more lenient)
        try:
            # Handle 'Z' suffix for UTC
            if v_stripped.endswith('Z'):
                v_stripped = v_stripped[:-1] + '+00:00'
            elif v_stripped.endswith('z'):
                v_stripped = v_stripped[:-1] + '+00:00'
            
            dt = datetime.fromisoformat(v_stripped)
            
            # Ensure dt is timezone-aware for comparison
            if dt.tzinfo is None:
                # If naive, assume UTC
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Warn if in past but allow it
            now_utc = datetime.now(timezone.utc)
            if dt < now_utc:
                print(f"[deferral.create] Warning: until_ISO is in the past: {v}")
            
            # Return original format if parseable
            return v.strip()
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {str(e)}. Expected ISO8601 format like '2025-11-06T10:00:00Z' or '2025-11-06T10:00:00+00:00'")

class DeferralCreateResponse(BaseModel):
    deferral_id: str
    next_contact_at: str  # ISO8601 datetime
    created_at: str  # ISO8601 datetime
    volunteer_id: str

@router.post("/deferral.create", response_model=DeferralCreateResponse)
async def create_deferral(req: DeferralCreateRequest) -> DeferralCreateResponse:
    """
    Create a deferral record when a volunteer wants to postpone onboarding.
    
    Supports idempotency: if idempotency_key is provided and a deferral exists,
    returns the existing deferral instead of creating a duplicate.
    
    Returns:
        DeferralCreateResponse with deferral_id, next_contact_at, created_at, volunteer_id
    """
    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Generate idempotency key if not provided
    idempotency_key = req.idempotency_key
    if not idempotency_key:
        idempotency_key = f"{req.volunteerId}_DEFERRAL_{int(datetime.now(timezone.utc).timestamp())}"
    
    # Check for existing deferral by idempotency_key
    if idempotency_key in _DEFERRAL_STORE:
        existing = _DEFERRAL_STORE[idempotency_key]
        print(f"[deferral.create] Found existing deferral for idempotency_key: {idempotency_key}")
        return DeferralCreateResponse(
            deferral_id=existing["deferral_id"],
            next_contact_at=existing["next_contact_at"],
            created_at=existing["created_at"],
            volunteer_id=existing["volunteer_id"]
        )
    
    # Check if volunteer already has a deferral (by volunteerId)
    if req.volunteerId in _DEFERRAL_BY_VOLUNTEER:
        existing_key = _DEFERRAL_BY_VOLUNTEER[req.volunteerId]
        existing = _DEFERRAL_STORE.get(existing_key)
        if existing:
            print(f"[deferral.create] Found existing deferral for volunteerId: {req.volunteerId}")
            # Update with new until_ISO if provided
            existing["next_contact_at"] = req.until_ISO
            _DEFERRAL_STORE[existing_key] = existing
            return DeferralCreateResponse(
                deferral_id=existing["deferral_id"],
                next_contact_at=req.until_ISO,
                created_at=existing["created_at"],
                volunteer_id=existing["volunteer_id"]
            )
    
    # Create new deferral
    deferral_id = f"defr_{uuid.uuid4().hex[:12]}"
    
    deferral_record = {
        "deferral_id": deferral_id,
        "next_contact_at": req.until_ISO,
        "created_at": now_iso,
        "volunteer_id": req.volunteerId,
        "reason": req.reason,
        "idempotency_key": idempotency_key
    }
    
    # Store by idempotency_key and volunteerId
    _DEFERRAL_STORE[idempotency_key] = deferral_record
    _DEFERRAL_BY_VOLUNTEER[req.volunteerId] = idempotency_key
    
    print(f"[deferral.create] Created deferral {deferral_id} for volunteer {req.volunteerId}, next contact: {req.until_ISO}")
    
    return DeferralCreateResponse(
        deferral_id=deferral_id,
        next_contact_at=req.until_ISO,
        created_at=now_iso,
        volunteer_id=req.volunteerId
    )

