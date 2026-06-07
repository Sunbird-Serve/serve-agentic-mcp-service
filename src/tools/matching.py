"""
Fulfilment MCP: Rule-Based Matching Engine
- Match volunteers to best-fit needs based on day/time preferences
- Score matches using weighted rules (day overlap, time overlap, recency)
- Return ranked results with score breakdown and reasons
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import httpx

from config import settings

router = APIRouter()

# --------- Constants ---------

SERVE_BASE_URL = settings.SERVE_BASE_URL
HTTP_TIMEOUT = 10.0

# Scoring weights (sum to 1.0)
WEIGHT_DAY_OVERLAP = 0.45
WEIGHT_TIME_OVERLAP = 0.35
WEIGHT_RECENCY = 0.20

# Day normalization map
DAY_NORMALIZE: Dict[str, str] = {
    "mon": "MONDAY", "monday": "MONDAY",
    "tue": "TUESDAY", "tuesday": "TUESDAY",
    "wed": "WEDNESDAY", "wednesday": "WEDNESDAY",
    "thu": "THURSDAY", "thursday": "THURSDAY",
    "fri": "FRIDAY", "friday": "FRIDAY",
    "sat": "SATURDAY", "saturday": "SATURDAY",
    "sun": "SUNDAY", "sunday": "SUNDAY",
}

# --------- Models ---------


class VolunteerPreferences(BaseModel):
    days: List[str] = Field(..., description="Preferred days, e.g. ['Mon','Tue','Wed']")
    time_windows: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Preferred time windows, e.g. [{'start':'08:00','end':'11:00'}]"
    )
    timezone: str = Field(default="Asia/Kolkata")


class MatchRequest(BaseModel):
    volunteerId: str = Field(..., description="Volunteer ID")
    preferences: Optional[VolunteerPreferences] = Field(
        None,
        description="Volunteer preferences. If not provided, fetched from preferences store."
    )
    maxResults: int = Field(default=5, ge=1, le=20, description="Maximum matches to return")
    minScore: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum score threshold")


class MatchReason(BaseModel):
    rule: str
    detail: str
    score: float


class MatchResult(BaseModel):
    needId: str
    title: str
    schoolName: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[MatchReason]
    matchedDays: List[str]
    matchedTimeSlots: List[Dict[str, str]]
    needDays: List[str]
    needStartDate: str
    needEndDate: str


class MatchResponse(BaseModel):
    volunteerId: str
    totalNeedsScanned: int
    matchesFound: int
    matches: List[MatchResult]


# --------- Helper Functions ---------


def _normalize_day(day: str) -> str:
    """Normalize day string to uppercase full name."""
    return DAY_NORMALIZE.get(day.strip().lower(), day.strip().upper())


def _time_to_minutes(time_str: str) -> int:
    """Convert HH:MM to minutes since midnight."""
    try:
        parts = time_str.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 0


def _time_overlap_minutes(start1: int, end1: int, start2: int, end2: int) -> int:
    """Calculate overlap in minutes between two time ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)


def _calculate_day_score(vol_days: List[str], need_days: List[str]) -> tuple[float, List[str]]:
    """Calculate day overlap score and matched days."""
    vol_normalized = {_normalize_day(d) for d in vol_days}
    need_normalized = {_normalize_day(d) for d in need_days}

    if not need_normalized:
        return 1.0, list(vol_normalized)

    matched = vol_normalized & need_normalized
    if not matched:
        return 0.0, []

    score = len(matched) / len(need_normalized)
    return min(score, 1.0), sorted(matched)


def _calculate_time_score(
    vol_windows: List[Dict[str, str]],
    need_slots: List[Dict[str, Any]]
) -> tuple[float, List[Dict[str, str]]]:
    """Calculate time overlap score and matched slots."""
    if not vol_windows or not need_slots:
        return 0.5, []

    matched_slots = []
    total_overlap = 0
    total_need_duration = 0

    for slot in need_slots:
        slot_start = _time_to_minutes(slot.get("startTime", "00:00"))
        slot_end = _time_to_minutes(slot.get("endTime", "23:59"))
        slot_duration = max(slot_end - slot_start, 1)
        total_need_duration += slot_duration

        best_overlap = 0
        for window in vol_windows:
            win_start = _time_to_minutes(window.get("start", "00:00"))
            win_end = _time_to_minutes(window.get("end", "23:59"))
            overlap = _time_overlap_minutes(win_start, win_end, slot_start, slot_end)
            best_overlap = max(best_overlap, overlap)

        if best_overlap > 0:
            total_overlap += best_overlap
            matched_slots.append({
                "day": slot.get("day", ""),
                "startTime": slot.get("startTime", ""),
                "endTime": slot.get("endTime", ""),
                "overlapMinutes": str(best_overlap)
            })

    if total_need_duration == 0:
        return 0.5, matched_slots

    score = total_overlap / total_need_duration
    return min(score, 1.0), matched_slots


def _calculate_recency_score(start_date: str, end_date: str) -> float:
    """Score based on how soon the need starts and whether it's still active."""
    now = datetime.now(timezone.utc)

    try:
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
    except ValueError:
        return 0.5

    if end_dt and end_dt.replace(tzinfo=timezone.utc) < now:
        return 0.0

    if start_dt:
        days_until = (start_dt.replace(tzinfo=timezone.utc) - now).days
        if days_until < 0:
            return 0.8
        elif days_until <= 7:
            return 1.0
        elif days_until <= 30:
            return 0.7
        elif days_until <= 90:
            return 0.4
        else:
            return 0.2

    return 0.5


def _score_need(
    vol_days: List[str],
    vol_windows: List[Dict[str, str]],
    need_item: Dict[str, Any]
) -> Optional[MatchResult]:
    """Score a single need against volunteer preferences."""
    need_days = need_item.get("days", [])
    time_slots = need_item.get("timeSlots", [])
    start_date = need_item.get("startDate", "")
    end_date = need_item.get("endDate", "")

    day_score, matched_days = _calculate_day_score(vol_days, need_days)
    time_score, matched_slots = _calculate_time_score(vol_windows, time_slots)
    recency_score = _calculate_recency_score(start_date, end_date)

    total_score = (
        WEIGHT_DAY_OVERLAP * day_score +
        WEIGHT_TIME_OVERLAP * time_score +
        WEIGHT_RECENCY * recency_score
    )

    reasons = []
    if day_score > 0:
        reasons.append(MatchReason(
            rule="day_overlap",
            detail=f"Matched {len(matched_days)} day(s): {', '.join(matched_days)}",
            score=day_score
        ))
    if time_score > 0:
        reasons.append(MatchReason(
            rule="time_overlap",
            detail=f"Time windows overlap with {len(matched_slots)} slot(s)",
            score=time_score
        ))
    if recency_score > 0:
        reasons.append(MatchReason(
            rule="recency",
            detail=f"Need starts {start_date or 'unknown'}, ends {end_date or 'unknown'}",
            score=recency_score
        ))

    return MatchResult(
        needId=need_item.get("needId", ""),
        title=need_item.get("title", ""),
        schoolName=need_item.get("schoolName", ""),
        score=round(total_score, 3),
        reasons=reasons,
        matchedDays=matched_days,
        matchedTimeSlots=matched_slots,
        needDays=need_days,
        needStartDate=start_date,
        needEndDate=end_date,
    )


async def _fetch_needs(page: int = 0, size: int = 20) -> List[Dict[str, Any]]:
    """Fetch needs from the Serve API."""
    url = f"{SERVE_BASE_URL}/api/v1/serve-need/need/"
    params = {"page": page, "size": size, "status": "Approved"}

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                items = data.get("content", data.get("items", data.get("data", [])))
            elif isinstance(data, list):
                items = data
            else:
                items = []

            return items
    except Exception as e:
        print(f"[matching] Warning: Failed to fetch needs: {e}")
        return []


def _map_api_need_to_simple(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw API response item to simplified structure for scoring."""
    need = item.get("need", {})
    entity = item.get("entity", {})
    occurrence = item.get("occurrence", {})
    if not occurrence and isinstance(item.get("occurrences"), list) and item.get("occurrences"):
        occurrence = item["occurrences"][0]

    time_slots_data = item.get("timeSlots", []) or []
    time_slots = []
    for slot in time_slots_data:
        if isinstance(slot, dict):
            day = slot.get("day", "")
            start_time = slot.get("startTime", "") or slot.get("start_time", "")
            end_time = slot.get("endTime", "") or slot.get("end_time", "")
            if "T" in start_time:
                start_time = start_time.split("T")[1].split("+")[0][:5]
            if "T" in end_time:
                end_time = end_time.split("T")[1].split("+")[0][:5]
            if day and start_time and end_time:
                time_slots.append({"day": day.upper(), "startTime": start_time, "endTime": end_time})

    days = []
    for slot in time_slots_data:
        if isinstance(slot, dict) and slot.get("day"):
            d = slot["day"].upper()
            if d not in days:
                days.append(d)

    start_date = ""
    end_date = ""
    raw_start = occurrence.get("startDate") or need.get("startDate") or item.get("startDate") or ""
    raw_end = occurrence.get("endDate") or need.get("endDate") or item.get("endDate") or ""
    if raw_start and "T" in str(raw_start):
        start_date = str(raw_start).split("T")[0]
    elif raw_start:
        start_date = str(raw_start)[:10]
    if raw_end and "T" in str(raw_end):
        end_date = str(raw_end).split("T")[0]
    elif raw_end:
        end_date = str(raw_end)[:10]

    return {
        "needId": str(item.get("id") or need.get("id") or ""),
        "title": need.get("name") or need.get("title") or item.get("title") or "",
        "schoolName": entity.get("name") or "",
        "days": days,
        "timeSlots": time_slots,
        "startDate": start_date,
        "endDate": end_date,
    }


# --------- Endpoints ---------


@router.post("/fulfill.match", response_model=MatchResponse)
async def fulfill_match(req: MatchRequest) -> MatchResponse:
    """
    Match a volunteer to best-fit needs using rule-based scoring.

    Rules applied:
    - Day overlap (45%): How many of the need's required days match volunteer's preferred days
    - Time overlap (35%): How much the need's time slots overlap with volunteer's time windows
    - Recency (20%): How soon the need starts (favors needs starting within 7 days)

    Returns ranked matches sorted by score descending.
    """
    # Get volunteer preferences
    if req.preferences:
        vol_days = req.preferences.days
        vol_windows = [w if isinstance(w, dict) else {"start": "08:00", "end": "17:00"}
                       for w in req.preferences.time_windows]
    else:
        from tools.preferences import _PREFS
        stored = _PREFS.get(req.volunteerId)
        if stored and "prefs" in stored:
            prefs_data = stored["prefs"]
            vol_days = prefs_data.get("days", [])
            raw_windows = prefs_data.get("time_windows", [])
            vol_windows = []
            for w in raw_windows:
                if isinstance(w, dict):
                    vol_windows.append({"start": w.get("start", "08:00"), "end": w.get("end", "17:00")})
                else:
                    vol_windows.append({"start": "08:00", "end": "17:00"})
        else:
            raise HTTPException(
                status_code=422,
                detail="No preferences found. Provide preferences in request or save them first via preferences.save."
            )

    # Fetch all approved needs (paginate through them)
    all_needs_raw = []
    for page in range(3):  # Max 3 pages = 60 needs
        page_items = await _fetch_needs(page=page, size=20)
        if not page_items:
            break
        all_needs_raw.extend(page_items)

    # Map and score
    matches: List[MatchResult] = []
    for raw_item in all_needs_raw:
        simple = _map_api_need_to_simple(raw_item)
        if not simple["needId"]:
            continue
        result = _score_need(vol_days, vol_windows, simple)
        if result and result.score >= req.minScore:
            matches.append(result)

    # Sort by score descending
    matches.sort(key=lambda m: m.score, reverse=True)
    top_matches = matches[: req.maxResults]

    return MatchResponse(
        volunteerId=req.volunteerId,
        totalNeedsScanned=len(all_needs_raw),
        matchesFound=len(top_matches),
        matches=top_matches,
    )
