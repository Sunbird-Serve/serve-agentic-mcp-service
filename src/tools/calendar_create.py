import random, string, time
from pydantic import BaseModel, Field
from fastapi import APIRouter

router = APIRouter()

class CreateEventIn(BaseModel):
    title: str
    start_iso: str  # e.g. "2025-09-20T18:00:00+05:30"
    end_iso: str
    attendees: list[str] = Field(default_factory=list)
    timezone: str = "Asia/Kolkata"
    notes: str | None = None

class CreateEventOut(BaseModel):
    ok: bool
    event_id: str
    meeting_url: str

def _fake_meet_code():
    def chunk(n): return "".join(random.choices(string.ascii_lowercase, k=n))
    return f"https://meet.google.com/{chunk(3)}-{chunk(4)}-{chunk(3)}"

@router.post("/calendar.create_event", response_model=CreateEventOut)
async def calendar_create_event(body: CreateEventIn):
    # STUB: no real calendar call
    time.sleep(0.2)
    return CreateEventOut(
        ok=True,
        event_id="evt_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=10)),
        meeting_url=_fake_meet_code(),
    )
