"""
Serve Volunteer Status Tool
- Get volunteer status from SERVE system
- Returns onboarding status, nomination history, and activity summary
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx

from config import settings

router = APIRouter()

SERVE_BASE_URL = settings.SERVE_BASE_URL
HTTP_TIMEOUT = settings.SERVE_TIMEOUT_SECONDS

# --------- Models ---------


class VolunteerStatusRequest(BaseModel):
    volunteerId: str = Field(..., description="SERVE volunteer osid")


class NominationInfo(BaseModel):
    needId: str = ""
    needTitle: str = ""
    status: str = ""
    nominatedAt: str = ""


class VolunteerStatusResponse(BaseModel):
    found: bool
    volunteerId: str
    status: str = ""
    onboardStatus: str = ""
    name: str = ""
    email: str = ""
    nominations: List[NominationInfo] = Field(default_factory=list)
    profile_completion: str = ""
    message: str = ""


# --------- Helpers ---------


def _extract_onboard_status(profile_data: Dict[str, Any]) -> str:
    """Extract onboard status from profile response."""
    onboard_details = profile_data.get("onboardDetails", {})
    if isinstance(onboard_details, dict):
        statuses = onboard_details.get("onboardStatus", [])
        if isinstance(statuses, list) and statuses:
            last = statuses[-1]
            if isinstance(last, dict):
                return f"{last.get('onboardStep', '')}:{last.get('status', '')}"
    return ""


def _extract_nominations(fulfillments: List[Dict[str, Any]]) -> List[NominationInfo]:
    """Extract nomination info from fulfillment data."""
    nominations = []
    for f in fulfillments:
        if not isinstance(f, dict):
            continue
        nominations.append(NominationInfo(
            needId=str(f.get("needId") or f.get("need", {}).get("id", "")),
            needTitle=f.get("need", {}).get("name", "") if isinstance(f.get("need"), dict) else "",
            status=f.get("status", ""),
            nominatedAt=str(f.get("createdAt") or f.get("created_at") or ""),
        ))
    return nominations


# --------- Endpoint ---------


@router.post("/serve.volunteer.status", response_model=VolunteerStatusResponse)
async def get_volunteer_status(req: VolunteerStatusRequest) -> VolunteerStatusResponse:
    """
    Get comprehensive volunteer status from SERVE system.

    Fetches user info, profile, and nomination history.
    """
    volunteer_id = req.volunteerId

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # Fetch user basic info
            user_url = f"{SERVE_BASE_URL}/api/v1/serve-volunteering/user/{volunteer_id}"
            user_resp = await client.get(user_url)

            if user_resp.status_code == 404:
                return VolunteerStatusResponse(
                    found=False,
                    volunteerId=volunteer_id,
                    message="Volunteer not found"
                )

            if user_resp.status_code != 200:
                return VolunteerStatusResponse(
                    found=False,
                    volunteerId=volunteer_id,
                    message=f"Failed to fetch volunteer: HTTP {user_resp.status_code}"
                )

            user_data = user_resp.json()

            # Extract basic info
            identity = user_data.get("identityDetails", {})
            contact = user_data.get("contactDetails", {})
            name = identity.get("fullname") or identity.get("name") or ""
            email = contact.get("email") or ""
            status = user_data.get("status") or ""

            # Try to fetch profile for onboard status
            onboard_status = ""
            profile_completion = ""
            try:
                profile_url = f"{SERVE_BASE_URL}/api/v1/serve-volunteering/user/user-profile/{volunteer_id}"
                profile_resp = await client.get(profile_url)
                if profile_resp.status_code == 200:
                    profile_data = profile_resp.json()
                    onboard_status = _extract_onboard_status(profile_data)
                    profile_completion = str(
                        profile_data.get("onboardDetails", {}).get("profileCompletion", "")
                    )
            except Exception:
                pass

            # Try to fetch nominations/fulfillments
            nominations: List[NominationInfo] = []
            try:
                fulfill_url = f"{SERVE_BASE_URL}/api/v1/serve-fulfill/nomination/volunteer/{volunteer_id}"
                fulfill_resp = await client.get(fulfill_url)
                if fulfill_resp.status_code == 200:
                    fulfill_data = fulfill_resp.json()
                    if isinstance(fulfill_data, list):
                        nominations = _extract_nominations(fulfill_data)
                    elif isinstance(fulfill_data, dict):
                        items = fulfill_data.get("content", fulfill_data.get("items", []))
                        if isinstance(items, list):
                            nominations = _extract_nominations(items)
            except Exception:
                pass

            return VolunteerStatusResponse(
                found=True,
                volunteerId=volunteer_id,
                status=status,
                onboardStatus=onboard_status,
                name=name,
                email=email,
                nominations=nominations,
                profile_completion=profile_completion,
                message="Volunteer found",
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout fetching volunteer status")
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Connection error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch volunteer status: {str(e)}")
