"""
Volunteer Profiles & Preferences MCP Tools
- Save, update, and retrieve volunteer profiles
- Full JSON schema with validation
- File-based persistence
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
import json
import os
import threading

router = APIRouter()

# --------- Configuration ---------

PROFILE_STORE_DIR = os.environ.get("PROFILE_STORE_DIR", "./profile_store")

# --------- In-Memory Store + Persistence ---------

_PROFILE_STORE: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_loaded = False


def _ensure_store_dir():
    os.makedirs(PROFILE_STORE_DIR, exist_ok=True)


def _profiles_file_path() -> str:
    return os.path.join(PROFILE_STORE_DIR, "profiles.jsonl")


def _load_from_disk():
    global _loaded
    if _loaded:
        return
    _loaded = True
    _ensure_store_dir()
    fpath = _profiles_file_path()
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        vid = record.get("volunteerId")
                        if vid:
                            _PROFILE_STORE[vid] = record
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[profile] Warning: Failed to load profiles: {e}")


def _persist_profile(volunteer_id: str, record: Dict[str, Any]):
    try:
        _ensure_store_dir()
        record["volunteerId"] = volunteer_id
        with open(_profiles_file_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[profile] Warning: Failed to persist profile: {e}")


# --------- JSON Schema for Profile ---------

class IdentityDetails(BaseModel):
    fullname: str = Field(..., description="Full name of the volunteer")
    gender: Optional[str] = Field(None, description="Gender")
    dob: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    nationality: Optional[str] = Field(None, description="Nationality")


class ContactDetails(BaseModel):
    email: Optional[str] = Field(None, description="Email address")
    mobile: Optional[str] = Field(None, description="Mobile/WhatsApp number (E.164)")
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = Field(default="India")


class TeachingPreferences(BaseModel):
    subjects: List[str] = Field(default_factory=list, description="Subjects the volunteer can teach")
    grades: List[str] = Field(default_factory=list, description="Grade levels (e.g., '5', '6-8')")
    languages: List[str] = Field(default_factory=list, description="Languages of instruction")
    days: List[str] = Field(default_factory=list, description="Preferred days (Mon, Tue, ...)")
    time_windows: List[Dict[str, str]] = Field(default_factory=list, description="Time windows [{start, end}]")
    timezone: str = Field(default="Asia/Kolkata")


class EligibilityInfo(BaseModel):
    age_years: Optional[int] = Field(None, ge=0, le=120)
    has_device: Optional[bool] = None
    weekly_commitment_hours: Optional[float] = Field(None, ge=0)
    language_comfort: Optional[str] = None


class ProfileData(BaseModel):
    identity: Optional[IdentityDetails] = None
    contact: Optional[ContactDetails] = None
    teaching: Optional[TeachingPreferences] = None
    eligibility: Optional[EligibilityInfo] = None
    consent_given: Optional[bool] = None
    onboarding_status: Optional[str] = None
    notes: Optional[str] = None


# --------- Request/Response Models ---------


class ProfileSaveRequest(BaseModel):
    volunteerId: str = Field(..., description="Unique volunteer ID")
    profile: ProfileData = Field(..., description="Profile data to save")


class ProfileSaveResponse(BaseModel):
    saved: bool
    volunteerId: str
    version: int
    updated_at: str


class ProfileUpdateRequest(BaseModel):
    volunteerId: str = Field(..., description="Volunteer ID")
    updates: Dict[str, Any] = Field(..., description="Partial profile fields to update (deep merge)")


class ProfileUpdateResponse(BaseModel):
    updated: bool
    volunteerId: str
    version: int
    updated_at: str
    changed_fields: List[str]


class ProfileGetRequest(BaseModel):
    volunteerId: str = Field(..., description="Volunteer ID")


class ProfileGetResponse(BaseModel):
    found: bool
    volunteerId: str
    profile: Optional[Dict[str, Any]] = None
    version: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# --------- Helper Functions ---------


def _deep_merge(base: Dict, updates: Dict) -> Dict:
    """Deep merge updates into base dict."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_changed_fields(old: Dict, new: Dict, prefix: str = "") -> List[str]:
    """Get list of top-level fields that changed."""
    changed = []
    all_keys = set(list(old.keys()) + list(new.keys()))
    for key in all_keys:
        full_key = f"{prefix}{key}" if prefix else key
        if key not in old:
            changed.append(full_key)
        elif key not in new:
            changed.append(full_key)
        elif old[key] != new[key]:
            changed.append(full_key)
    return changed


# --------- Endpoints ---------


@router.post("/profile.save", response_model=ProfileSaveResponse)
async def profile_save(req: ProfileSaveRequest) -> ProfileSaveResponse:
    """
    Save a volunteer profile (creates or overwrites).

    Use this for initial profile creation or full replacement.
    For partial updates, use profile.update instead.
    """
    with _lock:
        _load_from_disk()

    now = datetime.now(timezone.utc).isoformat()
    version = 1

    if req.volunteerId in _PROFILE_STORE:
        version = _PROFILE_STORE[req.volunteerId].get("version", 0) + 1

    record = {
        "volunteerId": req.volunteerId,
        "profile": req.profile.model_dump(exclude_none=True),
        "version": version,
        "created_at": _PROFILE_STORE.get(req.volunteerId, {}).get("created_at", now),
        "updated_at": now,
    }

    with _lock:
        _PROFILE_STORE[req.volunteerId] = record

    _persist_profile(req.volunteerId, record)

    return ProfileSaveResponse(
        saved=True,
        volunteerId=req.volunteerId,
        version=version,
        updated_at=now,
    )


@router.post("/profile.update", response_model=ProfileUpdateResponse)
async def profile_update(req: ProfileUpdateRequest) -> ProfileUpdateResponse:
    """
    Partially update a volunteer profile (deep merge).

    Only the provided fields are updated; existing fields are preserved.
    Creates a new profile if one doesn't exist.
    """
    with _lock:
        _load_from_disk()

    now = datetime.now(timezone.utc).isoformat()

    if req.volunteerId in _PROFILE_STORE:
        existing = _PROFILE_STORE[req.volunteerId]
        old_profile = existing.get("profile", {})
        new_profile = _deep_merge(old_profile, req.updates)
        version = existing.get("version", 0) + 1
        changed = _get_changed_fields(old_profile, new_profile)
        created_at = existing.get("created_at", now)
    else:
        new_profile = req.updates
        version = 1
        changed = list(req.updates.keys())
        created_at = now

    record = {
        "volunteerId": req.volunteerId,
        "profile": new_profile,
        "version": version,
        "created_at": created_at,
        "updated_at": now,
    }

    with _lock:
        _PROFILE_STORE[req.volunteerId] = record

    _persist_profile(req.volunteerId, record)

    return ProfileUpdateResponse(
        updated=True,
        volunteerId=req.volunteerId,
        version=version,
        updated_at=now,
        changed_fields=changed,
    )


@router.post("/profile.get", response_model=ProfileGetResponse)
async def profile_get(req: ProfileGetRequest) -> ProfileGetResponse:
    """
    Retrieve a volunteer's full profile.

    Returns all stored profile data including teaching preferences,
    eligibility info, and contact details.
    """
    with _lock:
        _load_from_disk()

    if req.volunteerId not in _PROFILE_STORE:
        return ProfileGetResponse(found=False, volunteerId=req.volunteerId)

    record = _PROFILE_STORE[req.volunteerId]
    return ProfileGetResponse(
        found=True,
        volunteerId=req.volunteerId,
        profile=record.get("profile"),
        version=record.get("version", 0),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    )
