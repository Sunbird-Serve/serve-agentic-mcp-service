"""
Preference Confirmation Polisher
- Generates a single WhatsApp-ready line confirming parsed day/time prefs
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List

from tools.llm_core import call_llm_for_text

router = APIRouter()

class TimeWindow(BaseModel):
    start: str  # HH:MM
    end: str    # HH:MM

class PrefsPolishRequest(BaseModel):
    days: List[str] = Field(description="ISO day codes e.g., MON,WED,SAT")
    time_windows: List[TimeWindow]
    timezone: str = Field(description="IANA TZ e.g., Asia/Kolkata")
    weekend_gate: bool = Field(description="If weekends are limited")

class PrefsPolishResponse(BaseModel):
    line: str

@router.post("/prefs.confirmation_polish", response_model=PrefsPolishResponse)
async def prefs_confirmation_polish(req: PrefsPolishRequest) -> PrefsPolishResponse:
    days_csv = ", ".join(req.days)
    windows_str = ", ".join([f"{tw.start}-{tw.end}" for tw in req.time_windows])
    weekend_note = "Weekends are limited here, so I’ll prioritize weekdays." if req.weekend_gate and any(d in ("SAT","SUN") for d in req.days) else ""
    system = (
        "You are Sia. Rewrite given structured preferences into a warm, one-line WhatsApp confirmation. "
        "Mention local time (IST if Asia/Kolkata). If weekends present and weekend_gate=true, add a short note about weekends being limited. "
        "Keep it short and friendly; plain text; no emojis unless natural."
    )
    user = (
        f"days={req.days}\n"
        f"time_windows={windows_str}\n"
        f"timezone={req.timezone}\n"
        f"weekend_gate={str(req.weekend_gate).lower()}\n"
    )
    prompt = f"System:\n{system}\n\nUser:\n{user}\nOutput: one line only."
    text, error = await call_llm_for_text(prompt=prompt, temperature=0.2, max_tokens=80)
    if error or not text:
        # Fallback simple template
        line = f"I’ve noted {days_csv} ({windows_str} {('IST' if req.timezone=='Asia/Kolkata' else req.timezone)}). {weekend_note}".strip()
        return PrefsPolishResponse(line=line)
    line = text.strip()
    # Ensure single line and append weekend note if missing and required
    if "weekend" not in line.lower() and weekend_note:
        line = f"{line} {weekend_note}".strip()
    return PrefsPolishResponse(line=line)


