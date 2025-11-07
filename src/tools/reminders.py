from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class ReminderCreate(BaseModel):
    when_ISO: str
    reason: str
    volunteerId: Optional[str] = None

class OkResponse(BaseModel):
    ok: bool

@router.post("/reminder.create", response_model=OkResponse)
async def reminder_create(req: ReminderCreate) -> OkResponse:
    return OkResponse(ok=True)
