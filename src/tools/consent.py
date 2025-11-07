from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from datetime import datetime
import uuid

router = APIRouter()

_CONSENT_STORE: Dict[str, Dict] = {}

class ConsentRecord(BaseModel):
    volunteerId: str
    consentGiven: bool

class ConsentResponse(BaseModel):
    consent_id: str
    volunteer_id: str
    consent_given: bool
    recorded_at: str  # ISO8601 datetime

@router.post("/consent.record", response_model=ConsentResponse)
async def consent_record(req: ConsentRecord) -> ConsentResponse:
    """Record volunteer consent for onboarding"""
    now_iso = datetime.utcnow().isoformat() + "Z"
    consent_id = f"cons_{uuid.uuid4().hex[:12]}"
    
    _CONSENT_STORE[req.volunteerId] = {
        "consent_id": consent_id,
        "consent": req.consentGiven,
        "recordedAt": now_iso
    }
    
    return ConsentResponse(
        consent_id=consent_id,
        volunteer_id=req.volunteerId,
        consent_given=req.consentGiven,
        recorded_at=now_iso
    )
