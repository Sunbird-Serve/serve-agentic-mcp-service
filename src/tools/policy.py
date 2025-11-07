from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter()

class SchedulingPolicyRequest(BaseModel):
    region_id: Optional[str] = Field(default=None, description="Optional region identifier")

class SchedulingPolicyResponse(BaseModel):
    weekend_gate: bool
    blackout_dates: List[str]
    policy_version: str

@router.post("/policy.scheduling", response_model=SchedulingPolicyResponse)
async def policy_scheduling(req: SchedulingPolicyRequest) -> SchedulingPolicyResponse:
    # Static policy for now; region-specific logic can be added later
    return SchedulingPolicyResponse(
        weekend_gate=True,
        blackout_dates=[],
        policy_version="v1.0"
    )


