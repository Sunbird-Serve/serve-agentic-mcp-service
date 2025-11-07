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
    
    # Calculate end time for examples
    example_start = now + timedelta(days=1, hours=7)  # Tomorrow 7 PM
    example_end = example_start + timedelta(minutes=dur_min)
    
    prompt = f"""Parse time/date from natural language into structured slots.

CURRENT TIME: {now_date} {now_time} ({now_day})
TIMEZONE: {tz_offset_fmt}
MEETING DURATION: {dur_min} minutes

USER INPUT: "{text}"

TIME OF DAY MAPPINGS:
- morning = 10:00
- afternoon = 15:00  
- evening = 19:00
- night = 21:00

RULES:
1. Calculate correct future date:
   - "tomorrow" = {(now + timedelta(days=1)).strftime('%Y-%m-%d')}
   - "saturday" = next Saturday from {now_date}
   - "sunday" = next Sunday from {now_date}
   - "next week" = add 7 days

2. Combine date + time:
   - "tomorrow evening" = {(now + timedelta(days=1)).strftime('%Y-%m-%d')} 19:00
   - "saturday morning" = next Saturday 10:00

3. Create end_iso by adding {dur_min} minutes to start_iso

4. Format label as: "DDD DD MMM HH:MM AM/PM" (e.g., "Fri 11 Oct 07:00 PM")

5. Must be at least 2 hours after current time ({now.isoformat()})

EXAMPLE OUTPUT:
{{
  "slots": [
    {{
      "start_iso": "{example_start.isoformat()}",
      "end_iso": "{example_end.isoformat()}",
      "label": "{example_start.strftime('%a %d %b %I:%M %p')}",
      "confidence": 0.85
    }}
  ]
}}

Return ONLY the JSON object. No markdown, no explanation."""

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
