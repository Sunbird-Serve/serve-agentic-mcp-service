from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict

router = APIRouter()

class TelemetryEvent(BaseModel):
    event: str
    payload: Dict

class OkResponse(BaseModel):
    ok: bool

@router.post("/telemetry.emit", response_model=OkResponse)
async def telemetry_emit(req: TelemetryEvent) -> OkResponse:
    return OkResponse(ok=True)
