from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta

router = APIRouter()

# In-memory stores
_SLOTS: Dict[str, Dict] = {}          # slotId -> {day,start,end,label}
_HOLDS: Dict[str, Dict] = {}          # holdId -> {slotId, expiresAt}

class ProposeRequest(BaseModel):
    volunteerId: str
    timeBand: Optional[str] = Field(default=None, description="'8-11' | '12-15' | 'MORNING' | 'AFTERNOON'")
    daysWhitelist: Optional[List[str]] = None  # Mon..Sun (any form accepted)
    limit: int = 2
    seedTimeIso: Optional[str] = Field(default=None, description="Optional seed time to center proposals around")
    seedTimesIso: Optional[List[str]] = Field(default=None, description="Optional list of seed times to propose directly")
    tz: Optional[str] = Field(default="Asia/Kolkata", description="Timezone for seedTimeIso parsing")

class SlotItem(BaseModel):
    id: str
    day: str
    start: str
    end: str
    label: str
    startISO: Optional[str] = None
    endISO: Optional[str] = None

class ProposeResponse(BaseModel):
    slots: List[SlotItem]

@router.post("/slots.propose", response_model=ProposeResponse)
async def slots_propose(req: ProposeRequest) -> ProposeResponse:
    # Days filter: if None, interpret as no filter (use all days Mon..Sun)
    raw_days = req.daysWhitelist
    # Normalize incoming day names (accept fri/FRI/Friday, etc.)
    day_aliases = {
        "mon": "Mon", "monday": "Mon",
        "tue": "Tue", "tues": "Tue", "tuesday": "Tue",
        "wed": "Wed", "wednesday": "Wed",
        "thu": "Thu", "thur": "Thu", "thurs": "Thu", "thursday": "Thu",
        "fri": "Fri", "friday": "Fri",
        "sat": "Sat", "saturday": "Sat",
        "sun": "Sun", "sunday": "Sun",
    }
    if raw_days:
        days_norm: List[str] = []
        for d in raw_days:
            key = (d or "").strip().lower()
            canon = day_aliases.get(key)
            if canon and canon not in days_norm:
                days_norm.append(canon)
        days = days_norm or ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    else:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Normalize band labels
    band_key = (req.timeBand or "").strip() if req.timeBand else ""
    if band_key:
        uk = band_key.upper()
        if uk == "MORNING":
            band_key = "8-11"
        elif uk == "AFTERNOON":
            band_key = "12-15"

    band_defaults = {
        "8-11": [("08:30","09:00"),("10:00","10:30")],
        "12-15": [("12:30","13:00"),("14:30","15:00")]
    }

    # If seedTimeIso provided, infer band from seed and prefer windows near seed
    seeded_times: Optional[List[tuple[str,str]]] = None
    seed_dt: Optional[datetime] = None
    if req.seedTimeIso:
        try:
            # Try to parse ISO; assume timezone-aware input
            seed_dt = datetime.fromisoformat(req.seedTimeIso.replace('Z','+00:00'))
            hour = seed_dt.hour
            if 8 <= hour < 12:
                band_key = "8-11"
                # Center proposals around seed (seed, seed+60m)
                sh = f"{hour:02d}:{seed_dt.minute:02d}"
                seeded_times = [(sh, (seed_dt.replace(minute=(seed_dt.minute+30)%60, hour=hour + (1 if seed_dt.minute+30>=60 else 0))).strftime("%H:%M")), ("10:00","10:30")]
            elif 12 <= hour <= 23:
                band_key = "12-15"
                # Prefer around the seed hour for unrestricted times (seed, seed+30m)
                sh = f"{hour:02d}:{seed_dt.minute:02d}"
                # Compute end as +30m
                end_dt = seed_dt + timedelta(minutes=30)
                seeded_times = [(sh, end_dt.strftime("%H:%M"))]
        except Exception:
            seeded_times = None

    # If multiple seeds provided, propose directly for those seed times
    if req.seedTimesIso:
        out_multi: List[SlotItem] = []
        seen_ids: set[str] = set()
        for iso in req.seedTimesIso:
            try:
                sd = datetime.fromisoformat((iso or "").replace('Z','+00:00'))
            except Exception:
                continue
            # Filter by daysWhitelist if provided
            day_code_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
            day_code = day_code_map.get(sd.weekday(), "")
            if raw_days and day_code not in days:
                continue
            end_dt = sd + timedelta(minutes=30)
            label = sd.strftime("%a %d %b %I:%M %p") + " – " + end_dt.strftime("%I:%M %p")
            sid = f"{day_code}-{sd.strftime('%Y%m%dT%H%M')}"
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            _SLOTS[sid] = {"day": day_code, "start": sd.strftime("%H:%M"), "end": end_dt.strftime("%H:%M"), "label": label, "startISO": sd.isoformat(), "endISO": end_dt.isoformat()}
            out_multi.append(SlotItem(id=sid, day=day_code, start=sd.strftime("%H:%M"), end=end_dt.strftime("%H:%M"), label=label, startISO=sd.isoformat(), endISO=end_dt.isoformat()))
        # Sort and cap by limit
        out_multi.sort(key=lambda s: s.startISO or "")
        return ProposeResponse(slots=out_multi[: max(req.limit, 1)])

    times = seeded_times or band_defaults.get(band_key, band_defaults["12-15"])
    out: List[SlotItem] = []
    count = 0

    # Helper: map day code to weekday index
    day_to_idx = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    idx_to_day = {v: k for k, v in day_to_idx.items()}

    # If seed date present and no explicit day filter, center proposals on seed weekday forward
    ordered_days = list(days)
    if seed_dt and (req.daysWhitelist is None or len(req.daysWhitelist) == 0):
        seed_idx = seed_dt.weekday()  # Mon=0
        # Build ordered list starting from seed day
        ordered_days = []
        for offset in range(7):
            idx = (seed_idx + offset) % 7
            day_code = idx_to_day.get(idx)
            if day_code and day_code in days and day_code not in ordered_days:
                ordered_days.append(day_code)

    for d in ordered_days:
        for st, en in times:
            sid = f"{d}-{st}-{en}"
            start_iso = None
            end_iso = None
            label = f"{d} {st} – {en}"

            if seed_dt:
                # Compute next occurrence date for target weekday d
                target_idx = day_to_idx.get(d, seed_dt.weekday())
                delta_days = (target_idx - seed_dt.weekday()) % 7
                target_date = seed_dt.date() if delta_days == 0 else (seed_dt + timedelta(days=delta_days)).date()
                # Compose datetime using target_date and HH:MM
                sh, sm = [int(x) for x in st.split(":")]
                eh, em = [int(x) for x in en.split(":")]
                # Preserve timezone offset if present on seed
                tzinfo = seed_dt.tzinfo
                start_dt = datetime(target_date.year, target_date.month, target_date.day, sh, sm, 0, 0, tzinfo=tzinfo)
                end_dt = datetime(target_date.year, target_date.month, target_date.day, eh, em, 0, 0, tzinfo=tzinfo)
                start_iso = start_dt.isoformat()
                end_iso = end_dt.isoformat()
                label = start_dt.strftime("%a %d %b %I:%M %p") + " – " + end_dt.strftime("%I:%M %p")

            _SLOTS[sid] = {"day": d, "start": st, "end": en, "label": label, "startISO": start_iso, "endISO": end_iso}
            out.append(SlotItem(id=sid, day=d, start=st, end=en, label=label, startISO=start_iso, endISO=end_iso))
            count += 1
            if count >= req.limit:
                break
        if count >= req.limit:
            break
    return ProposeResponse(slots=out)

class HoldRequest(BaseModel):
    slotId: str

class HoldResponse(BaseModel):
    holdId: str
    expiresAt_ISO: str

@router.post("/slot.hold", response_model=HoldResponse)
async def slot_hold(req: HoldRequest) -> HoldResponse:
    if req.slotId not in _SLOTS:
        raise HTTPException(status_code=404, detail="slot_not_found")
    holdId = f"h_{req.slotId}_{int(datetime.utcnow().timestamp())}"
    expires = datetime.utcnow() + timedelta(minutes=2)
    _HOLDS[holdId] = {"slotId": req.slotId, "expiresAt": expires}
    return HoldResponse(holdId=holdId, expiresAt_ISO=expires.isoformat())

class BookRequest(BaseModel):
    holdId: str

class BookResponse(BaseModel):
    meetingUrl: str
    startISO: str
    endISO: str

@router.post("/slot.book", response_model=BookResponse)
async def slot_book(req: BookRequest) -> BookResponse:
    if req.holdId not in _HOLDS:
        raise HTTPException(status_code=404, detail="hold_not_found")
    hold = _HOLDS[req.holdId]
    if hold["expiresAt"] < datetime.utcnow():
        raise HTTPException(status_code=409, detail="hold_expired")
    slot = _SLOTS.get(hold["slotId"]) or {}
    # If proposal carried startISO/endISO, use them; else fallback to naive today-based times
    start_iso = slot.get("startISO")
    end_iso = slot.get("endISO")
    if not start_iso or not end_iso:
        today = datetime.utcnow()
        start_iso = today.replace(hour=int(slot.get("start","12:30").split(':')[0]), minute=int(slot.get("start","12:30").split(':')[1]), second=0, microsecond=0).isoformat()
        end_iso = today.replace(hour=int(slot.get("end","13:00").split(':')[0]), minute=int(slot.get("end","13:00").split(':')[1]), second=0, microsecond=0).isoformat()
    return BookResponse(meetingUrl="https://meet.example.com/serve", startISO=start_iso, endISO=end_iso)
