from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

# Import stores if available
try:
    from .consent import _CONSENT_STORE
except Exception:
    _CONSENT_STORE = {}
try:
    from .preferences import _PREFS
except Exception:
    _PREFS = {}

class ProfileGet(BaseModel):
    volunteerId: str

class ProfileResponse(BaseModel):
    profile: Dict[str, Any]

@router.post("/profile.get", response_model=ProfileResponse)
async def profile_get(req: ProfileGet) -> ProfileResponse:
    profile: Dict[str, Any] = {}
    if req.volunteerId in _CONSENT_STORE:
        profile["consent"] = _CONSENT_STORE[req.volunteerId]
    if req.volunteerId in _PREFS:
        profile["preferences"] = _PREFS[req.volunteerId]
    return ProfileResponse(profile=profile)
