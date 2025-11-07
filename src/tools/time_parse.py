import re
import json
from fastapi import APIRouter, Request
from pydantic import BaseModel
from datetime import datetime, timedelta
import pytz
from .models import Slot  # Shared model
from .llm_time_parse import _llm_parse_time  # Import LLM parser from separate tool

router = APIRouter()

# ------------- MODEL DEFINITIONS -------------

class ParseRequest(BaseModel):
    text: str
    tz: str = "Asia/Kolkata"
    duration_minutes: int = 30

class ParseResponse(BaseModel):
    slots: list[Slot]
    needs_clarification: bool = False
    reason: str | None = None

# ------------- CONFIG -------------

MAX_OPTIONS = 3
ENABLE_LLM = True  # Set to False to disable LLM fallback (fast parser only)


# ------------- FAST PARSER (direct times like Fri 9AM, 10 Oct 7PM) -------------

def _fast_parse(text: str, tz: str, duration_minutes: int) -> tuple[list[Slot], list[str]]:
    """
    Returns: (parsed_slots, unparsed_parts)
    - parsed_slots: list of successfully parsed time slots
    - unparsed_parts: list of text segments that couldn't be parsed (for LLM fallback)
    """
    import dateparser
    now = datetime.now(pytz.timezone(tz))
    parts = re.split(r"\bor\b|,|/|;", text)
    slots = []
    unparsed = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue

        dt = dateparser.parse(
            part, 
            settings={
                "TIMEZONE": tz, 
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",  # Always interpret as future dates
                "PREFER_DAY_OF_MONTH": "first",  # Prefer upcoming occurrence
                "RELATIVE_BASE": now  # Calculate relative to now
            }
        )
        
        # If parsed date is in the past, try to find next occurrence
        if dt and dt < now:
            # For day names (Monday, Tuesday, etc.), add 7 days to get next week
            day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if any(day in part.lower() for day in day_names):
                dt = dt + timedelta(days=7)
                print(f"[time_parse] Adjusted '{part}' from past to future: {dt.strftime('%Y-%m-%d %H:%M')}")
        if not dt:
            unparsed.append(part)
            continue
        if dt < now + timedelta(hours=1):  # must be at least an hour later
            print(f"[time_parse] Rejected '{part}': too soon ({dt.strftime('%Y-%m-%d %H:%M')})")
            unparsed.append(part)
            continue
        end = dt + timedelta(minutes=duration_minutes)
        label = dt.strftime("%a %d %b %I:%M %p")
        print(f"[time_parse] Parsed '{part}' -> {label} ({dt.isoformat()})")
        slots.append(Slot(start_iso=dt.isoformat(), end_iso=end.isoformat(), label=label))
    
    return slots[:MAX_OPTIONS], unparsed


# LLM fallback is now a separate MCP tool (llm_time_parse.py)
# We import and use _llm_parse_time from that module


# ------------- MAIN ENDPOINT -------------

@router.post("/time.parse_options", response_model=ParseResponse)
async def parse_options(req: ParseRequest):
    tz = req.tz or "Asia/Kolkata"
    now = datetime.now(pytz.timezone(tz))

    print(f"[time.parse_options] incoming text='{req.text}' tz={tz}")
    
    # Try fast parser first
    fast_slots, unparsed_parts = _fast_parse(req.text, tz, req.duration_minutes)
    
    if fast_slots:
        print(f"[time.parse_options] FAST-PATH parsed {len(fast_slots)} slot(s)")
    
    # If there are unparsed parts, send them to LLM (if enabled)
    all_slots = list(fast_slots)  # Start with fast parser results
    
    if unparsed_parts and ENABLE_LLM:
        unparsed_text = " or ".join(unparsed_parts)
        print(f"[time.parse_options] {len(unparsed_parts)} unparsed part(s), calling llm.parse_time: {unparsed_text}")
        llm_slots = await _llm_parse_time(unparsed_text, tz, req.duration_minutes, now)
        
        if llm_slots:
            print(f"[time.parse_options] LLM parsed {len(llm_slots)} additional slot(s)")
            all_slots.extend(llm_slots)
        else:
            print("[time.parse_options] LLM returned 0 slots for unparsed parts")
    elif unparsed_parts and not ENABLE_LLM:
        print(f"[time.parse_options] {len(unparsed_parts)} unparsed part(s) - LLM disabled")
    
    # If we have any slots (from either parser), return them
    if all_slots:
        # Sort by start time and deduplicate
        all_slots.sort(key=lambda s: s.start_iso)
        # Remove duplicates based on start_iso
        seen = set()
        unique_slots = []
        for slot in all_slots:
            if slot.start_iso not in seen:
                seen.add(slot.start_iso)
                unique_slots.append(slot)
        
        return ParseResponse(slots=unique_slots[:MAX_OPTIONS], needs_clarification=False)
    
    # No slots parsed at all - ask for clarification
    print("[time.parse_options] No slots parsed from any method")
    return ParseResponse(
        slots=[],
        needs_clarification=True,
        reason="Please share explicit times (e.g., 'Fri 7 pm', 'Sat 8:30 pm', '10 Oct 8pm'). You can send multiple using 'or'."
    )
