"""
LLM Time Parser - Specialized tool using generic LLM core
"""
import json
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta
import pytz
from typing import Optional, List
from .models import Slot  # Shared model
from .llm_core import call_llm_for_json, extract_json_from_text

router = APIRouter()

# ------------- MODEL DEFINITIONS -------------

class LLMParseRequest(BaseModel):
    text: str
    tz: str = "Asia/Kolkata"
    duration_minutes: int = 30
    now_iso: Optional[str] = None  # Optional: current time override for testing

class LLMParseResponse(BaseModel):
    slots: List[Slot]
    success: bool = True
    error: Optional[str] = None

# ------------- LLM TIME PARSER -------------

async def _llm_parse_time(text: str, tz: str, dur_min: int, now: datetime) -> List[Slot]:
    """
    Use LLM to parse vague/natural language time expressions.
    Examples: "tomorrow evening", "saturday morning", "next week monday afternoon"
    """
    # Format current time info
    now_date = now.strftime("%Y-%m-%d")
    now_day = now.strftime("%A")
    now_time = now.strftime("%H:%M")
    tz_offset = now.strftime("%z")
    tz_offset_fmt = f"{tz_offset[:3]}:{tz_offset[3:]}"  # +0530 -> +05:30
    
    prompt = (
        f"Current datetime: {now_date} {now_time} ({now_day}), timezone offset: {tz_offset_fmt}\n"
        f"User request: {json.dumps(text)}\n"
        f"Meeting duration: {dur_min} minutes\n\n"
        "Instructions:\n"
        "1. Parse the date and time from user request\n"
        "2. Time of day meanings: morning=10:00, afternoon=15:00, evening=19:00, night=21:00\n"
        "3. Calculate the correct date (if 'tomorrow', add 1 day; if 'saturday', find next Saturday, etc.)\n"
        "4. Must be at least 2 hours after current datetime\n"
        "5. Return JSON in this exact format:\n"
        f'{{"slots":[{{"start_iso":"YYYY-MM-DDTHH:MM:SS{tz_offset_fmt}","end_iso":"YYYY-MM-DDTHH:MM:SS{tz_offset_fmt}","label":"DDD DD MMM HH:MM AM/PM","confidence":0.8}}]}}\n\n'
        "Return only the JSON, no other text."
    )

    print(f"[llm.parse_time] Calling LLM core for: {text}")
    
    # Use generic LLM core
    data, error = await call_llm_for_json(
        prompt=prompt,
        temperature=0,  # Deterministic for date/time
        max_tokens=200,
        context_window=1536,
        threads=4
    )
    
    if error:
        print(f"[llm.parse_time] LLM core failed: {error}")
        return []
    
    if not data:
        print("[llm.parse_time] No data returned from LLM")
        return []
    
    # Parse slots from response
    slots_out = []
    for s in (data.get("slots") or []):
        try:
            slots_out.append(Slot(**s))
        except Exception as e:
            print(f"[llm.parse_time] Failed to parse slot: {e}")
            continue
    
    print(f"[llm.parse_time] Successfully parsed {len(slots_out)} slot(s)")
    return slots_out

# ------------- MCP ENDPOINT -------------

@router.post("/llm.parse_time", response_model=LLMParseResponse)
async def llm_parse_time(req: LLMParseRequest):
    """
    MCP Tool: llm.parse_time
    
    Uses LLM to parse natural language time expressions into structured slots.
    Built on top of generic llm.core.
    
    Examples:
    - "tomorrow evening" -> slot at tomorrow 7 PM
    - "saturday morning" -> slot at next Saturday 10 AM
    - "next week monday afternoon" -> slot at next Monday 3 PM
    """
    tz = pytz.timezone(req.tz)
    now = (
        datetime.fromisoformat(req.now_iso).astimezone(tz)
        if req.now_iso else datetime.now(tz)
    )
    
    print(f"[llm.parse_time] Request: text='{req.text}' tz={req.tz}")
    
    slots = await _llm_parse_time(req.text, req.tz, req.duration_minutes, now)
    
    if not slots:
        return LLMParseResponse(
            slots=[],
            success=False,
            error="Could not parse time from text. LLM returned no results."
        )
    
    return LLMParseResponse(slots=slots, success=True)
