"""
Time Refinement Tool - Allow users to modify/refine parsed time slots
"""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional
import pytz
from .models import Slot  # Shared model
from .time_parse import _fast_parse
from .llm_time_parse import _llm_parse_time

router = APIRouter()

# ------------- MODEL DEFINITIONS -------------

class RefineRequest(BaseModel):
    original_slots: List[Slot]  # Previously returned slots
    refinement_text: str  # User's refinement request
    tz: str = "Asia/Kolkata"
    duration_minutes: int = 30

class RefineResponse(BaseModel):
    refined_slots: List[Slot]
    action_taken: str  # "replaced", "added", "removed", "modified"
    message: str  # Human-readable explanation
    original_count: int
    refined_count: int

# ------------- PARSE REFINE (NEW) -------------

class PolicyWindow(BaseModel):
    start: str  # "HH:MM"
    end: str    # "HH:MM"

class ParsePolicy(BaseModel):
    weekday_only: bool
    window_24h: PolicyWindow
    map_phrases: bool

class ParseRefineRequest(BaseModel):
    text: str
    policy: ParsePolicy
    need_at_least: int = 2
    locale: str = "en-IN"

class SimpleSlot(BaseModel):
    day: str  # Mon..Fri
    start: str  # HH:MM
    end: str    # HH:MM
    confidence: float

class ParseRefineResponse(BaseModel):
    slots: List[SimpleSlot]
    notes: str | None = None

_DAY_MAP = {
    "mon": "Mon", "monday": "Mon",
    "tue": "Tue", "tues": "Tue", "tuesday": "Tue",
    "wed": "Wed", "wednesday": "Wed",
    "thu": "Thu", "thur": "Thu", "thurs": "Thu", "thursday": "Thu",
    "fri": "Fri", "friday": "Fri",
}

def _clamp_time(hhmm: str, window: PolicyWindow) -> str:
    return hhmm  # simplified: assume mapped windows are within bounds

def _phrase_windows(text: str) -> list[tuple[str, str, float, str]]:
    """Return list of (start, end, conf, note) for known phrases."""
    t = text.lower()
    out: list[tuple[str, str, float, str]] = []
    if any(p in t for p in ["lunch", "noon"]):
        out.append(("12:30", "13:00", 0.9, "Mapped lunch/noon -> 12:30–13:00"))
    if any(p in t for p in ["morning", "first half"]):
        out.append(("08:00", "11:00", 0.8, "Mapped morning/first half -> 08:00–11:00"))
    if any(p in t for p in ["afternoon", "post-lunch", "post lunch"]):
        out.append(("14:00", "15:00", 0.8, "Mapped afternoon/post-lunch -> 14:00–15:00"))
    # explicitly exclude evening
    return out

@router.post("/time.parser_refine", response_model=ParseRefineResponse)
async def time_parser_refine(req: ParseRefineRequest) -> ParseRefineResponse:
    """
    Parse vague day/time phrases into constrained weekday slots.
    Errors are returned as empty slots with notes; never throws.
    """
    try:
        text = (req.text or "").strip()
        notes: list[str] = []
        slots: list[SimpleSlot] = []

        # Detect days mentioned
        import re
        mentioned_days: list[str] = []
        for key, out_day in _DAY_MAP.items():
            if re.search(rf"\b{key}\b", text, re.I):
                if out_day not in mentioned_days:
                    mentioned_days.append(out_day)

        if not mentioned_days and req.policy.weekday_only:
            mentioned_days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

        # Map phrases to windows
        windows: list[tuple[str, str, float, str]] = []
        if req.policy.map_phrases:
            windows = _phrase_windows(text)
            for _, _, _, n in windows:
                notes.append(n)

        # If no phrase windows, default to policy window
        if not windows:
            w = req.policy.window_24h
            windows = [(w.start, w.end, 0.6, "Defaulted to policy window")]

        # Build slots combining days and windows
        for day in mentioned_days:
            for start, end, conf, _ in windows:
                slots.append(SimpleSlot(day=day, start=_clamp_time(start, req.policy.window_24h), end=_clamp_time(end, req.policy.window_24h), confidence=conf))
                if len(slots) >= max(req.need_at_least, 1):
                    break
            if len(slots) >= max(req.need_at_least, 1):
                break

        # If still not enough, pad across remaining weekdays
        if len(slots) < req.need_at_least:
            remaining_days = [d for d in ["Mon","Tue","Wed","Thu","Fri"] if d not in mentioned_days]
            for day in remaining_days:
                for start, end, conf, _ in windows:
                    slots.append(SimpleSlot(day=day, start=start, end=end, confidence=conf * 0.9))
                    if len(slots) >= req.need_at_least:
                        break
                if len(slots) >= req.need_at_least:
                    break

        return ParseRefineResponse(slots=slots, notes="; ".join(notes) if notes else None)
    except Exception as e:
        return ParseRefineResponse(slots=[], notes=f"Failed to parse: {type(e).__name__}")

# ------------- REFINEMENT LOGIC -------------

def _detect_refinement_intent(text: str) -> tuple[str, str]:
    """
    Detect what the user wants to do.
    Returns: (intent, normalized_text)
    
    Intents:
    - "replace_all": "no, I meant..." / "change to..." / "instead..."
    - "replace_one": "not friday, make it saturday" / "change friday to saturday"
    - "add": "also add..." / "include..." / "and..."
    - "remove": "remove friday" / "not friday" / "cancel friday"
    - "shift": "later" / "earlier" / "delay by 1 hour" / "advance by 30 min"
    """
    lower = text.lower().strip()
    
    # Replace all
    if any(phrase in lower for phrase in ["no,", "instead", "change to", "i meant", "actually"]):
        # Extract what comes after the trigger phrase
        for phrase in ["no,", "instead", "change to", "i meant", "actually"]:
            if phrase in lower:
                idx = lower.index(phrase) + len(phrase)
                return "replace_all", text[idx:].strip()
    
    # Replace one specific slot
    if "not " in lower and (" make it " in lower or " change to " in lower):
        return "replace_one", text
    
    if "change " in lower and " to " in lower:
        return "replace_one", text
    
    # Remove
    if any(phrase in lower for phrase in ["remove ", "cancel ", "delete ", "drop "]):
        return "remove", text
    
    # Add
    if any(phrase in lower for phrase in ["also ", "add ", "include ", "plus ", " and ", ", "]):
        return "add", text
    
    # Shift time
    if any(phrase in lower for phrase in ["later", "earlier", "delay", "advance", "shift", "move"]):
        return "shift", text
    
    # Default: treat as replace all
    return "replace_all", text

def _extract_slot_identifier(text: str, slots: List[Slot]) -> Optional[int]:
    """
    Try to identify which slot the user is referring to.
    Returns index of the slot, or None if can't determine.
    """
    lower = text.lower()
    
    # Try to match by day of week
    for i, slot in enumerate(slots):
        dt = datetime.fromisoformat(slot.start_iso)
        day_name = dt.strftime("%A").lower()
        day_abbr = dt.strftime("%a").lower()
        
        if day_name in lower or day_abbr in lower:
            return i
    
    # Try to match by date
    for i, slot in enumerate(slots):
        dt = datetime.fromisoformat(slot.start_iso)
        date_str = dt.strftime("%d").lstrip("0")  # "10" or "5"
        if f" {date_str} " in f" {lower} " or f" {date_str}th" in lower or f" {date_str}st" in lower:
            return i
    
    # Try to match by position ("first", "second", "last")
    if "first" in lower or "1st" in lower:
        return 0
    if ("second" in lower or "2nd" in lower) and len(slots) > 1:
        return 1
    if "last" in lower and slots:
        return len(slots) - 1
    
    return None

def _extract_time_shift(text: str) -> Optional[timedelta]:
    """
    Extract time shift amount from text.
    Examples: "1 hour later", "30 minutes earlier", "delay by 2 hours"
    Returns: timedelta (positive for later, negative for earlier)
    """
    import re
    lower = text.lower()
    
    # Find number and unit
    match = re.search(r'(\d+)\s*(hour|hr|minute|min)', lower)
    if not match:
        # Default shifts
        if "later" in lower:
            return timedelta(hours=1)
        if "earlier" in lower:
            return timedelta(hours=-1)
        return None
    
    amount = int(match.group(1))
    unit = match.group(2)
    
    if "hour" in unit or unit == "hr":
        delta = timedelta(hours=amount)
    else:  # minutes
        delta = timedelta(minutes=amount)
    
    # Direction
    if any(word in lower for word in ["earlier", "before", "advance"]):
        delta = -delta
    
    return delta

# ------------- MAIN REFINEMENT FUNCTION -------------

async def _refine_slots(
    original_slots: List[Slot],
    refinement_text: str,
    tz: str,
    duration_minutes: int
) -> tuple[List[Slot], str, str]:
    """
    Process refinement request and return modified slots.
    Returns: (refined_slots, action_taken, message)
    """
    intent, normalized_text = _detect_refinement_intent(refinement_text)
    
    # REPLACE ALL - user wants completely different time(s)
    if intent == "replace_all":
        # Try fast parser first
        fast_slots, unparsed = _fast_parse(normalized_text, tz, duration_minutes)
        
        if fast_slots:
            return (
                fast_slots,
                "replaced",
                f"Replaced all slots with new time(s): {normalized_text}"
            )
        
        # Try LLM for vague expressions
        if unparsed or not fast_slots:
            now = datetime.now(pytz.timezone(tz))
            llm_slots = await _llm_parse_time(normalized_text, tz, duration_minutes, now)
            
            if llm_slots:
                return (
                    llm_slots,
                    "replaced",
                    f"Replaced all slots with new time(s): {normalized_text}"
                )
        
        # Couldn't parse new time
        return (
            original_slots,
            "unchanged",
            f"Couldn't understand the new time '{normalized_text}'. Original slots kept."
        )
    
    # REPLACE ONE - user wants to change a specific slot
    elif intent == "replace_one":
        slot_idx = _extract_slot_identifier(refinement_text, original_slots)
        
        if slot_idx is None:
            return (
                original_slots,
                "unchanged",
                "Couldn't identify which slot to replace. Please be more specific."
            )
        
        # Parse the new time
        now = datetime.now(pytz.timezone(tz))
        fast_slots, _ = _fast_parse(normalized_text, tz, duration_minutes)
        
        if not fast_slots:
            llm_slots = await _llm_parse_time(normalized_text, tz, duration_minutes, now)
            fast_slots = llm_slots
        
        if fast_slots:
            new_slots = original_slots.copy()
            old_label = new_slots[slot_idx].label
            new_slots[slot_idx] = fast_slots[0]
            
            return (
                new_slots,
                "modified",
                f"Changed '{old_label}' to '{fast_slots[0].label}'"
            )
        
        return (
            original_slots,
            "unchanged",
            f"Couldn't parse the new time from: {normalized_text}"
        )
    
    # ADD - user wants to add more slot(s)
    elif intent == "add":
        now = datetime.now(pytz.timezone(tz))
        fast_slots, unparsed = _fast_parse(normalized_text, tz, duration_minutes)
        
        all_new_slots = list(fast_slots)
        
        if unparsed:
            unparsed_text = " or ".join(unparsed)
            llm_slots = await _llm_parse_time(unparsed_text, tz, duration_minutes, now)
            all_new_slots.extend(llm_slots)
        
        if all_new_slots:
            combined = original_slots + all_new_slots
            # Deduplicate by start_iso
            seen = set()
            unique = []
            for slot in combined:
                if slot.start_iso not in seen:
                    seen.add(slot.start_iso)
                    unique.append(slot)
            
            unique.sort(key=lambda s: s.start_iso)
            
            return (
                unique,
                "added",
                f"Added {len(all_new_slots)} more slot(s) to the existing ones"
            )
        
        return (
            original_slots,
            "unchanged",
            f"Couldn't parse additional times from: {normalized_text}"
        )
    
    # REMOVE - user wants to remove a slot
    elif intent == "remove":
        slot_idx = _extract_slot_identifier(refinement_text, original_slots)
        
        if slot_idx is None:
            return (
                original_slots,
                "unchanged",
                "Couldn't identify which slot to remove. Please be more specific."
            )
        
        if len(original_slots) == 1:
            return (
                original_slots,
                "unchanged",
                "Cannot remove the only remaining slot. Please provide alternative time(s) instead."
            )
        
        removed_slot = original_slots[slot_idx]
        new_slots = [s for i, s in enumerate(original_slots) if i != slot_idx]
        
        return (
            new_slots,
            "removed",
            f"Removed '{removed_slot.label}'"
        )
    
    # SHIFT - user wants to shift time(s)
    elif intent == "shift":
        shift_delta = _extract_time_shift(refinement_text)
        
        if shift_delta is None:
            return (
                original_slots,
                "unchanged",
                "Couldn't understand the time shift. Please specify like '1 hour later' or '30 minutes earlier'."
            )
        
        # Check if shifting specific slot or all
        slot_idx = _extract_slot_identifier(refinement_text, original_slots)
        
        if slot_idx is not None:
            # Shift specific slot
            new_slots = original_slots.copy()
            old_slot = new_slots[slot_idx]
            old_dt = datetime.fromisoformat(old_slot.start_iso)
            new_dt = old_dt + shift_delta
            new_end = new_dt + timedelta(minutes=duration_minutes)
            
            new_slots[slot_idx] = Slot(
                start_iso=new_dt.isoformat(),
                end_iso=new_end.isoformat(),
                label=new_dt.strftime("%a %d %b %I:%M %p"),
                confidence=old_slot.confidence
            )
            
            return (
                new_slots,
                "modified",
                f"Shifted '{old_slot.label}' by {shift_delta}"
            )
        else:
            # Shift all slots
            new_slots = []
            for slot in original_slots:
                old_dt = datetime.fromisoformat(slot.start_iso)
                new_dt = old_dt + shift_delta
                new_end = new_dt + timedelta(minutes=duration_minutes)
                
                new_slots.append(Slot(
                    start_iso=new_dt.isoformat(),
                    end_iso=new_end.isoformat(),
                    label=new_dt.strftime("%a %d %b %I:%M %p"),
                    confidence=slot.confidence
                ))
            
            return (
                new_slots,
                "modified",
                f"Shifted all slots by {shift_delta}"
            )
    
    # Fallback
    return (
        original_slots,
        "unchanged",
        "Couldn't understand the refinement request. Please try: 'change to...', 'add...', 'remove...', 'shift later/earlier'."
    )

# ------------- MCP ENDPOINT -------------

@router.post("/time.refine_slots", response_model=RefineResponse)
async def refine_slots(req: RefineRequest):
    """
    MCP Tool: time.refine_slots
    
    Allows users to refine/modify previously suggested time slots.
    
    Supported operations:
    1. Replace all: "no, I meant friday 7pm" / "change to saturday morning"
    2. Replace one: "not friday, make it saturday" / "change first slot to 8pm"
    3. Add: "also add sunday 10am" / "include monday afternoon"
    4. Remove: "remove friday" / "cancel the first one"
    5. Shift: "1 hour later" / "30 minutes earlier" / "shift friday to 2 hours later"
    
    Examples:
    - User gets: [Fri 5pm, Sat 10am]
      Says: "no, instead give me sunday evening"
      Result: [Sun 7pm]
    
    - User gets: [Fri 5pm, Sat 10am]
      Says: "remove friday"
      Result: [Sat 10am]
    
    - User gets: [Fri 5pm]
      Says: "also add saturday 3pm"
      Result: [Fri 5pm, Sat 3pm]
    
    - User gets: [Fri 5pm]
      Says: "make it 1 hour later"
      Result: [Fri 6pm]
    """
    print(f"[time.refine_slots] Original: {len(req.original_slots)} slots, Refinement: '{req.refinement_text}'")
    
    refined_slots, action, message = await _refine_slots(
        req.original_slots,
        req.refinement_text,
        req.tz,
        req.duration_minutes
    )
    
    print(f"[time.refine_slots] Action: {action}, Result: {len(refined_slots)} slots")
    
    return RefineResponse(
        refined_slots=refined_slots,
        action_taken=action,
        message=message,
        original_count=len(req.original_slots),
        refined_count=len(refined_slots)
    )

