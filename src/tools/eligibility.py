from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter()

class EligibilityRequest(BaseModel):
    ageYears: int
    hasDevice: bool
    weeklyCommitmentHours: Optional[float] = Field(default=None, ge=0)

class EligibilityResponse(BaseModel):
    eligible: bool
    reasons: Optional[List[str]] = None

@router.post("/eligibility.check", response_model=EligibilityResponse)
async def eligibility_check(req: EligibilityRequest) -> EligibilityResponse:
    """
    Check volunteer eligibility.
    
    For ELIGIBILITY_PART1: Only age and device are required (weeklyCommitmentHours can be null).
    For ELIGIBILITY_PART2: All three fields required (weeklyCommitmentHours must be >= 2.0).
    """
    reasons: List[str] = []
    if req.ageYears < 18:
        reasons.append("under_18")
    if not req.hasDevice:
        reasons.append("no_device")
    # Only check commitment if provided (for PART2)
    if req.weeklyCommitmentHours is not None:
        if req.weeklyCommitmentHours < 2:
            reasons.append("insufficient_commitment")
    return EligibilityResponse(eligible=len(reasons) == 0, reasons=reasons or None)
