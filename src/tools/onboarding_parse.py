"""
Onboarding natural-language parser: extracts intents, consent, constraints, availability, and eligibility hints.
- Phrase mapping (lunch/morning/afternoon), weekday-only policy
- Returns confidences, never throws; empty/low confidence when unclear
- Optional LLM enhancement for ambiguous cases
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import re

from .llm_core import call_llm_for_json

router = APIRouter()

DayLiteral = Literal["Mon","Tue","Wed","Thu","Fri"]
ConsentLiteral = Literal["yes","no","defer","unknown"]

class ParseRequest(BaseModel):
    text: str
    locale: str = "en-IN"
    state: Optional[str] = None  # Optional: Current onboarding state for context-aware parsing

class AvailabilityItem(BaseModel):
    day: DayLiteral
    start: str
    end: str
    confidence: float = Field(ge=0.0, le=1.0)
    # Optional parse-friendly fields so client can map bands reliably
    explicit_hour: Optional[str] = None  # HH:MM representative time
    start_iso: Optional[str] = None  # Prefer when explicit date phrases are present (e.g., today/tomorrow)

class Consent(BaseModel):
    value: ConsentLiteral
    confidence: float = Field(ge=0.0, le=1.0)

class Constraints(BaseModel):
    weekday_ok: Optional[bool] = None
    weekend_only: Optional[bool] = None
    confidence: float = Field(ge=0.0, le=1.0)

class TimeWindow(BaseModel):
    start: str  # HH:MM
    end: str    # HH:MM

class EligibilityHints(BaseModel):
    age_ok: Optional[bool] = None
    age_years: Optional[int] = None
    device_ok: Optional[bool] = None
    has_device: Optional[bool] = None
    device_type: Optional[str] = None
    commitment_hint: Optional[str] = None
    weekly_commitment_hours: Optional[float] = None
    same_day_request: Optional[bool] = None
    confidence: float = Field(ge=0.0, le=1.0)

class ParseResponse(BaseModel):
    intents: List[str]
    consent: Consent
    constraints: Constraints
    availability: List[AvailabilityItem]
    eligibility: Optional[EligibilityHints] = None
    followup: Optional[str] = None
    # PREFS_DAYTIME outputs (optional)
    days: Optional[List[str]] = None
    time_windows: Optional[List[TimeWindow]] = None
    paraphrase: Optional[str] = None
    prefs_confidence: Optional[float] = None

_DAY_MAP = {
    "mon": "Mon", "monday": "Mon",
    "tue": "Tue", "tues": "Tue", "tuesday": "Tue",
    "wed": "Wed", "wednesday": "Wed",
    "thu": "Thu", "thur": "Thu", "thurs": "Thu", "thursday": "Thu",
    "fri": "Fri", "friday": "Fri",
}

def _detect_intents(t: str) -> List[str]:
    intents: List[str] = []
    low = t.lower()
    if any(w in low for w in ["yes", "sure", "okay", "ok", "sounds good", "works"]):
        intents.append("consent")
    if any(w in low for w in ["no", "not possible", "can't", "cant", "won't", "dont want"]):
        intents.append("decline")
    if any(w in low for w in ["think", "later", "get back", "maybe"]):
        intents.append("defer")
    if any(w in low for w in ["lunch", "noon", "morning", "first half", "afternoon", "post-lunch", "post lunch"]):
        intents.append("availability_hint")
    if any(w in low for w in ["saturday", "sunday", "weekend"]):
        intents.append("weekend_preference")
    if any(w in low for w in ["weekday", "week days", "school hours", "8", "15"]):
        intents.append("weekday_alignment")
    return intents or ["unknown"]

def _parse_consent(t: str) -> Consent:
    low = t.lower().strip()
    words = low.split()
    
    # Expanded yes patterns (including "fine", "alright")
    yes_patterns = ["yes", "yeah", "yep", "sure", "okay", "ok", "works", "ready", "fine", "alright", "correct", "agreed"]
    # Expanded no patterns
    no_patterns = ["no", "nope", "not possible", "can't", "cant", "won't", "cannot", "decline"]
    
    # For short responses (1-3 words), higher confidence
    if len(words) <= 3:
        if any(w in low for w in yes_patterns) and not any(n in low for n in ["no", "not", "can't", "cant"]):
            return Consent(value="yes", confidence=0.9)  # Higher confidence for clear short responses
        if any(w in low for w in no_patterns):
            return Consent(value="no", confidence=0.9)  # Higher confidence for clear short responses
    
    # For longer responses, still check patterns but lower confidence
    if any(w in low for w in yes_patterns) and not any(n in low for n in ["no", "not", "can't", "cant"]):
        return Consent(value="yes", confidence=0.8)
    if any(w in low for w in no_patterns):
        return Consent(value="no", confidence=0.8)
    if any(w in low for w in ["think", "later", "get back", "maybe"]):
        return Consent(value="defer", confidence=0.75)
    return Consent(value="unknown", confidence=0.4)

def _parse_constraints(t: str) -> Constraints:
    low = t.lower()
    # Affirmative responses (yes, ok, should be ok, etc.) when asked about weekday constraints
    affirmative = any(
        p in low for p in [
            "yes", "sure", "okay", "ok", "sounds good", "works", "should be ok",
            "should be fine", "that works", "fine", "alright", "agreed"
        ]
    ) and not any(n in low for n in ["no", "not", "can't", "cant", "won't"])
    
    # Explicit weekend-only expressions
    explicit_weekend_only = any(
        p in low for p in [
            "weekends only", "weekend only", "only weekends", "no weekday", "no weekdays",
            "can it be weekend", "over weekend", "on weekends", "prefer weekends"
        ]
    )
    # Negative toward weekend ("no weekend") means weekday_ok likely
    no_weekend = any(p in low for p in ["no weekend", "not weekend"])
    weekend = any(w in low for w in ["saturday", "sunday", "weekend"]) and not no_weekend
    weekday = any(w in low for w in ["weekday", "week days", "school hours", "8-15", "8 to 15", "8 am", "3 pm"]) or no_weekend

    # If affirmative and no weekend mention, assume weekday_ok
    if affirmative and not weekend:
        return Constraints(weekday_ok=True, weekend_only=False, confidence=0.85)
    
    if explicit_weekend_only:
        return Constraints(weekday_ok=False, weekend_only=True, confidence=0.9)
    if weekend and not weekday:
        return Constraints(weekday_ok=False, weekend_only=True, confidence=0.8)
    if weekday and not weekend:
        return Constraints(weekday_ok=True, weekend_only=False, confidence=0.8 if no_weekend else 0.7)
    if weekend and weekday:
        return Constraints(weekday_ok=True, weekend_only=False, confidence=0.6)
    return Constraints(confidence=0.4)

def _phrase_window(low: str) -> Optional[tuple[str,str,float,str]]:
    # Returns (start, end, confidence, explicit_hour)
    if any(p in low for p in ["lunch", "noon"]):
        return ("12:30","13:00",0.9,"12:30")
    if any(p in low for p in ["morning", "first half"]):
        return ("08:00","11:00",0.8,"09:30")
    if any(p in low for p in ["afternoon", "post-lunch", "post lunch"]):
        return ("14:00","15:00",0.8,"14:30")
    return None

def _parse_availability(t: str) -> List[AvailabilityItem]:
    low = t.lower()
    # exclude evening outright
    if "evening" in low:
        return []
    days: List[DayLiteral] = []
    for key, out_day in _DAY_MAP.items():
        if re.search(rf"\b{key}\b", low, re.I):
            if out_day not in days:
                days.append(out_day)  # type: ignore
    if not days:
        days = ["Mon","Tue","Wed","Thu","Fri"]  # type: ignore
    # Try to capture specific times first (e.g., 10 am, 18:30)
    explicit_hour: Optional[str] = None
    m = re.search(r"\b(\d{1,2})(:(\d{2}))?\s*(am|pm)\b", low)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(3) or 0)
        ampm = m.group(4)
        if ampm == "pm" and hh != 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        explicit_hour = f"{hh:02d}:{mm:02d}"
    else:
        m2 = re.search(r"\b(\d{2}):(\d{2})\b", low)
        if m2:
            explicit_hour = f"{int(m2.group(1)):02d}:{int(m2.group(2)):02d}"

    pw = _phrase_window(low)
    if pw:
        start, end, conf, phr_exp = pw
        if explicit_hour is None:
            explicit_hour = phr_exp
    else:
        start, end, conf = ("08:00","15:00",0.6)
    out: List[AvailabilityItem] = []
    for d in days:
        out.append(AvailabilityItem(day=d, start=start, end=end, confidence=conf, explicit_hour=explicit_hour))
    return out

def _parse_eligibility_hints(t: str, state: Optional[str] = None) -> EligibilityHints:
    low = t.lower()
    age_ok: Optional[bool] = True if any(w in low for w in ["i am 18", "i am 19", "i am 20", "i'm 18", "i'm 19", "i'm 20"]) else None
    age_years: Optional[int] = None
    age_patterns = [
        r"(?:i am|i'm|im|age is|i turned|turning)\s*(\d{1,2})",
        r"(\d{1,2})\s*(?:yrs?|years?|y/o|yo)\b",
    ]
    for pat in age_patterns:
        match = re.search(pat, low)
        if match:
            try:
                age_years = int(match.group(1))
                break
            except (TypeError, ValueError):
                continue
    if age_years is not None:
        if age_ok is None:
            age_ok = age_years >= 18
        else:
            # trust explicit number more if conflicting
            age_ok = age_years >= 18 or age_ok
    device_type: Optional[str] = None
    device_words = {
        "laptop": ["laptop", "macbook", "chromebook", "notebook"],
        "desktop": ["desktop", "pc"],
        "tablet": ["tablet", "ipad"],
        "smartphone": ["smartphone", "phone", "iphone", "android"],
    }
    device_ok = None
    for dtype, words in device_words.items():
        if any(word in low for word in words):
            device_type = dtype
            device_ok = True
            break
    if device_ok is None:
        device_ok = True if any(w in low for w in ["smartphone", "phone", "laptop", "tablet"]) else None
    has_device = device_ok
    weekly_hours: Optional[float] = None
    commitment_hint = None
    # Parse numeric hours, including ranges 2-3 hours
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?\s*(hours?|hrs?)", low)
    if m:
        try:
            a = float(m.group(1))
            b = float(m.group(2)) if m.group(2) else None
            weekly_hours = (a + b) / 2.0 if b else a
        except Exception:
            weekly_hours = None
        if weekly_hours is not None:
            commitment_hint = f"{weekly_hours} hours/week"
    # Heuristic: "yes for all", "yes for all three", "yes, all good"
    if any(p in low for p in ["yes for all", "yes for all three", "yes to all", "all yes", "all good", "yes for all 3", "yes for all three of them"]):
        age_ok = True
        device_ok = True
        has_device = True
        if weekly_hours is None:
            weekly_hours = 2.0
            commitment_hint = "2.0 hours/week"
    # State-aware short-affirmatives for ELIGIBILITY_PART1
    if (state or "").upper() == "ELIGIBILITY_PART1":
        # Device confirmation
        if any(p in low for p in ["i have", "i do have", "yes i have", "got one"]) and not device_ok:
            device_ok = True
            has_device = True
        # Age confirmation
        if any(p in low for p in ["i am", "yes i am", "i'm"]) and age_ok is None:
            age_ok = True
    signal_score = 0.0
    if age_years is not None:
        signal_score += 0.3
    if age_ok is not None:
        signal_score += 0.2
    if device_type:
        signal_score += 0.3
    elif device_ok is not None:
        signal_score += 0.2
    if weekly_hours is not None:
        signal_score += 0.2
    conf = min(0.95, 0.35 + signal_score) if signal_score > 0 else 0.3
    return EligibilityHints(
        age_ok=age_ok,
        age_years=age_years,
        device_ok=device_ok,
        has_device=has_device,
        device_type=device_type,
        weekly_commitment_hours=weekly_hours,
        commitment_hint=commitment_hint,
        confidence=conf
    )

async def _llm_enhance_parse(text: str, locale: str, rule_result: ParseResponse, state: Optional[str] = None) -> Optional[ParseResponse]:
    """
    Optional LLM enhancement when rule-based parsing has low confidence or ambiguous results.
    Returns enhanced ParseResponse if LLM succeeds, None otherwise.
    """
    # Check if we should use LLM: low confidence or ambiguous
    min_confidence = min(
        rule_result.consent.confidence,
        rule_result.constraints.confidence,
        rule_result.eligibility.confidence if rule_result.eligibility else 1.0
    )
    
    # Use LLM if overall confidence is low (< 0.5) or multiple intents detected (ambiguous)
    if min_confidence >= 0.5 and len(rule_result.intents) == 1 and rule_result.intents[0] != "unknown":
        return None  # Rule-based is confident enough
    
    state_context = state or "UNKNOWN"
    
    # Build state-aware prompt; if ELIGIBILITY_PART2, include commitment-specific policy and same-day detection
    if state_context == "ELIGIBILITY_PART2":
        prompt = f"""You are parsing a volunteer’s onboarding message for SERVE. Your job is to return structured fields for the current state without changing state or generating replies. Be precise, policy-safe, and extract numeric values when present.

CONTEXT
state: "{state_context}"
locale: "{locale}"
policy:
Weekday-only during school hours (08:00–15:00)
~2 hours per week required
Hours must be split across different weekdays (not all on the same day)
100% volunteer; no pay
rule_signals (validate and improve):
intents: {rule_result.intents}
eligibility: age_ok={rule_result.eligibility.age_ok if rule_result.eligibility else None}, device_ok={rule_result.eligibility.device_ok if rule_result.eligibility else None}, weekly_commitment_hours={rule_result.eligibility.weekly_commitment_hours if rule_result.eligibility else None}
User
"{text}"
Instructions (strict)
1) Extract weekly_commitment_hours as a number (float) whenever any hint exists (even with low certainty).
Normalize word numbers and variants:
“an hour”, “a hour”, “one hour”, “1 hr”, “1hour”, “1.5 hours”, “2hrs”, “two hours” → numeric float (1.0, 1.5, 2.0, etc.)
If user implies “less than 2 hours” without a number (e.g., “maybe an hour”), set weekly_commitment_hours = 1.0 with lower confidence.
If user implies “more than 2 hours” (e.g., “3 hours”), set weekly_commitment_hours = 3.0.
2) Detect same_day_request for phrasing like:
“same day”, “sameday”, “same-day”, “today”, “both hours today”, “2 hours in one day”, “two hours same day”
same_day_request: true if present; else false
3) Confidence
Always output weekly_commitment_hours as a number when any evidence exists; encode uncertainty in confidence (0.0–1.0), don’t omit the number because of low confidence.
If nothing at all about hours, set weekly_commitment_hours = null and confidence ~0.3.
4) Booleans
has_device and device_ok are synonyms; if you infer one, set both consistently.
If not mentioned, leave as null (not false).
5) Output only JSON with this schema (no extra text, no markdown):
{{
"intents": ["..."],
"eligibility": {{
"age_ok": true|false|null,
"device_ok": true|false|null,
"has_device": true|false|null,
"weekly_commitment_hours": number|null,
"same_day_request": true|false,
"confidence": 0.0
}}
}}
Few-shot guidance (do not echo; use to calibrate)
“I can give an hour maybe” → weekly_commitment_hours: 1.0, same_day_request: false, confidence ~0.6
“1 hr to start with” → weekly_commitment_hours: 1.0, same_day_request: false, confidence ~0.7
“Can I do 2 hours same day?” → weekly_commitment_hours: 2.0, same_day_request: true, confidence ~0.9
“Two and a half hours works” → weekly_commitment_hours: 2.5, same_day_request: false, confidence ~0.9
“Not sure, maybe some weeks” → weekly_commitment_hours: null, same_day_request: false, confidence ~0.4
Return ONLY the JSON object."""
    elif state_context == "ELIGIBILITY_PART1":
        prompt = f"""You are parsing a volunteer's onboarding message for SERVE. Return structured booleans for age and device without changing state or generating replies. Strict JSON only.

CONTEXT
state: "{state_context}"
locale: "{locale}"
rule_signals:
eligibility: age_ok={rule_result.eligibility.age_ok if rule_result.eligibility else None}, device_ok={rule_result.eligibility.device_ok if rule_result.eligibility else None}
USER
"{text}"
INSTRUCTIONS
1) Short affirmatives like "I do have", "Yes I have", "I have", "got one" → device_ok=true, has_device=true (confidence ~0.8–0.9).
2) Short affirmatives like "I am", "Yes I am", "I'm" → age_ok=true (confidence ~0.8–0.9).
3) Do not infer negative unless explicit (e.g., "don't have"). If unclear, set null and use lower confidence.
4) Output only:
{{
  "eligibility": {{
    "age_ok": true|false|null,
    "device_ok": true|false|null,
    "has_device": true|false|null,
    "confidence": 0.0
  }}
}}
EXAMPLES (do not echo)
"I do have" → device_ok: true, has_device: true, confidence: 0.9
"Yes I have" → device_ok: true, has_device: true, confidence: 0.9
"I have" → device_ok: true, has_device: true, confidence: 0.8
"I am" → age_ok: true, confidence: 0.9
"Yes I am" → age_ok: true, confidence: 0.9
Return ONLY the JSON object."""
    else:
        prompt = f"""You are parsing a volunteer's onboarding message. Extract structured information for the current onboarding state.

CONTEXT
state: "{state_context}" // e.g., WELCOME | ELIGIBILITY_PART1 | ELIGIBILITY_PART2
locale: "{locale}"
USER MESSAGE
"{text}"
RULE-BASED SIGNALS (validate and improve)
intents: {rule_result.intents}
consent: {rule_result.consent.value} ({rule_result.consent.confidence:.2f})
constraints: weekday_ok={rule_result.constraints.weekday_ok}, weekend_only={rule_result.constraints.weekend_only} ({rule_result.constraints.confidence:.2f})
eligibility: age_ok={rule_result.eligibility.age_ok if rule_result.eligibility else None}, device_ok={rule_result.eligibility.device_ok if rule_result.eligibility else None}, weekly_commitment_hours={rule_result.eligibility.weekly_commitment_hours if rule_result.eligibility else None}

EXTRACTION RULES (strict)
Always return numbers for weekly_commitment_hours (float), never strings.
Normalize word-numbers and variants:
"an hour", "a hour", "one hour", "1 hr", "1hour", "1.5 hours" → 1.0, 1.0, 1.0, 1.0, 1.0, 1.5
If user mentions a number with uncertainty ("maybe 1 hour", "start with an hour"), still extract that number and reflect uncertainty via confidence (e.g., 0.5–0.7).
If state = ELIGIBILITY_PART2, prioritize extracting weekly_commitment_hours from the message.
device_ok and has_device are synonymous; set both consistently when evident.
If no clear evidence, set fields to null (not false), and reduce confidence.

OUTPUT FORMAT (return ONLY this JSON)
{{
  "intents": ["..."],
  "consent": {{"value": "yes|no|defer|unknown", "confidence": 0.0}},
  "constraints": {{"weekday_ok": true|false|null, "weekend_only": true|false|null, "confidence": 0.0}},
  "eligibility": {{
    "age_ok": true|false|null,
    "device_ok": true|false|null,
    "has_device": true|false|null,
    "weekly_commitment_hours": number|null,
    "confidence": 0.0
  }}
}}

FEW EXAMPLES (for calibration; do not echo)
"I can give an hour maybe" → weekly_commitment_hours: 1.0, confidence ~0.6
"1 hr to start with" → weekly_commitment_hours: 1.0, confidence ~0.7
"Yes I can do two hours" → weekly_commitment_hours: 2.0, confidence ~0.9
"No time right now" → weekly_commitment_hours: null, confidence ~0.9

Return ONLY the JSON object."""

    data, error = await call_llm_for_json(
        prompt=prompt,
        temperature=0.2,
        max_tokens=300
    )
    
    if error or not data:
        return None  # LLM failed, use rule-based result
    
    # Merge LLM results (LLM takes precedence for ambiguous cases)
    try:
        # Validate consent
        consent_val = data.get("consent", {}).get("value", rule_result.consent.value)
        if consent_val not in ["yes", "no", "defer", "unknown"]:
            consent_val = rule_result.consent.value
        
        # Build enhanced response
        enhanced_consent = Consent(
            value=consent_val,
            confidence=data.get("consent", {}).get("confidence", rule_result.consent.confidence)
        )
        
        constraints_data = data.get("constraints", {})
        enhanced_constraints = Constraints(
            weekday_ok=constraints_data.get("weekday_ok", rule_result.constraints.weekday_ok),
            weekend_only=constraints_data.get("weekend_only", rule_result.constraints.weekend_only),
            confidence=constraints_data.get("confidence", rule_result.constraints.confidence)
        )
        
        eligibility_data = data.get("eligibility", {})
        enhanced_eligibility = EligibilityHints(
            age_ok=eligibility_data.get("age_ok", rule_result.eligibility.age_ok if rule_result.eligibility else None),
            age_years=eligibility_data.get("age_years", rule_result.eligibility.age_years if rule_result.eligibility else None),
            device_ok=eligibility_data.get("device_ok", rule_result.eligibility.device_ok if rule_result.eligibility else None),
            has_device=eligibility_data.get("has_device", rule_result.eligibility.has_device if rule_result.eligibility else None),
            device_type=eligibility_data.get("device_type", rule_result.eligibility.device_type if rule_result.eligibility else None),
            weekly_commitment_hours=eligibility_data.get("weekly_commitment_hours", rule_result.eligibility.weekly_commitment_hours if rule_result.eligibility else None),
            same_day_request=eligibility_data.get("same_day_request", getattr(rule_result.eligibility, 'same_day_request', None) if rule_result.eligibility else None),
            confidence=eligibility_data.get("confidence", rule_result.eligibility.confidence if rule_result.eligibility else 0.3)
        )
        
        # Determine followup
        followup = rule_result.followup
        if enhanced_consent.value == "defer":
            followup = "Shall I set a reminder to check in next week?"
        elif enhanced_constraints.weekend_only:
            followup = "Could you try a short weekday slot—perhaps a 20–30 min lunch break?"
        elif not rule_result.availability:
            followup = "Could you share any weekday times between 8 and 15 that might work?"
        
        return ParseResponse(
            intents=data.get("intents", rule_result.intents),
            consent=enhanced_consent,
            constraints=enhanced_constraints,
            availability=rule_result.availability,  # Keep rule-based availability
            eligibility=enhanced_eligibility,
            followup=followup
        )
    except Exception:
        return None  # Error parsing LLM response, use rule-based

@router.post("/onboarding.parse_message", response_model=ParseResponse)
async def onboarding_parse_message(req: ParseRequest) -> ParseResponse:
    try:
        text = (req.text or "").strip()
        # PREFS_DAYTIME state: run dedicated day/time extraction prompt
        if (req.state or "").upper() == "PREFS_DAYTIME":
            # Build prompt per provided spec
            policy_version = "v1.0"
            weekend_gate = True
            blackout_dates: List[str] = []
            prompt = f"""You are “Sia”, SERVE’s onboarding assistant. Extract structured day/time preferences from the user’s message. Do NOT schedule or promise anything. Return ONLY JSON.

Context
{{
"state": "PREFS_DAYTIME",
"timezone": "{req.locale if req.locale else 'Asia/Kolkata'}",
"policy_version": "{policy_version}",
"weekend_gate": {str(weekend_gate).lower()},
"blackout_dates": {blackout_dates}
}}
User
"{text}"
Instructions (strict)
Parse days to ISO codes: ["MON","TUE","WED","THU","FRI","SAT","SUN"].
Accept informal terms and expand:
“weekdays” → MON..FRI (keep first three if you cap)
“weekends” → SAT,SUN
Parse time bands:
mornings → 08:00–11:00
afternoons/noon → 12:00–16:00
evenings → 17:00–20:00
If a specific time is given (e.g., “6pm”, “18:30”), create a reasonable window around it (90–120 min) within school hours if possible.
If user lists many days, prefer the first three mentioned (unless your policy stores all).
If weekend_gate=true and SAT/SUN are mentioned, keep them in the JSON but do not remove or alter them. (Client will add the “weekends limited” note.)
If the user does NOT mention any time-of-day band (morning/afternoon/evening) or any specific time (e.g., 6pm, 18:30), then set time_windows = [] (do not infer a broad default window).
Return a concise paraphrase of what you extracted, and a numeric confidence 0–1.
Output (strict JSON)
{{
"days": ["MON","WED"],
"time_windows": [{{"start":"17:00","end":"20:00"}}],
"confidence": 0.0,
"paraphrase": "Evenings on Mondays and Wednesdays."
}}
"""
            data, err = await call_llm_for_json(prompt=prompt, temperature=0.2, max_tokens=300)
            days: List[str] = []
            windows: List[TimeWindow] = []
            confidence = 0.3
            paraphrase = ""
            if not err and isinstance(data, dict):
                days = data.get("days") or []
                tw = data.get("time_windows") or []
                try:
                    windows = [TimeWindow(**w) for w in tw if isinstance(w, dict) and w.get("start") and w.get("end")]
                except Exception:
                    windows = []
                confidence = float(data.get("confidence") or 0.3)
                paraphrase = (data.get("paraphrase") or "").strip()
            # Server-side guard: if no time clues in text, do not infer broad defaults
            lower_txt = text.lower()
            has_band = any(k in lower_txt for k in ["morning", "mornings", "afternoon", "noon", "evening", "evenings"])
            import re as _re
            has_specific = bool(_re.search(r"\b(\d{1,2})(:?\d{2})?\s*(am|pm)\b|\b\d{2}:\d{2}\b", lower_txt))
            if not has_band and not has_specific:
                windows = []
            # Return structured-only; keep other fields minimal
            return ParseResponse(
                intents=["unknown"],
                consent=Consent(value="unknown", confidence=0.2),
                constraints=Constraints(confidence=0.2),
                availability=[],
                eligibility=None,
                days=days,
                time_windows=windows,
                paraphrase=paraphrase,
                prefs_confidence=confidence
            )
        intents = _detect_intents(text)
        consent = _parse_consent(text)
        constraints = _parse_constraints(text)
        availability = _parse_availability(text)
        eligibility = _parse_eligibility_hints(text, req.state)

        followup: Optional[str] = None
        if consent.value == "defer":
            followup = "Shall I set a reminder to check in next week?"
        elif constraints.weekend_only:
            followup = "Could you try a short weekday slot—perhaps a 20–30 min lunch break?"
        elif not availability:
            followup = "Could you share any weekday times between 8 and 15 that might work?"

        # Create rule-based result
        rule_result = ParseResponse(
            intents=intents,
            consent=consent,
            constraints=constraints,
            availability=availability,
            eligibility=eligibility,
            followup=followup
        )
        
        # Step 2: Optional LLM enhancement for ambiguous/low-confidence cases
        # Only use LLM if rule-based confidence is low or multiple intents (ambiguous)
        # Don't penalize for low eligibility confidence if we have clear consent/constraints
        primary_confidence = min(consent.confidence, constraints.confidence)
        min_confidence = min(
            primary_confidence,
            eligibility.confidence if eligibility else 1.0
        )
        is_ambiguous = len(intents) > 1 or "unknown" in intents
        
        # Only use LLM if:
        # 1. Primary fields (consent/constraints) are unclear (< 0.6), OR
        # 2. Multiple intents or unknown intent, OR  
        # 3. All fields have low confidence (< 0.5)
        use_llm = (primary_confidence < 0.6 or is_ambiguous or min_confidence < 0.5)
        
        if use_llm:
            print(f"[onboarding.parse_message] Low confidence ({min_confidence:.2f}) or ambiguous, trying LLM enhancement")
            enhanced = await _llm_enhance_parse(text, req.locale, rule_result, req.state)
            if enhanced:
                print(f"[onboarding.parse_message] LLM enhancement successful")
                return enhanced
            else:
                print(f"[onboarding.parse_message] LLM enhancement failed or skipped, using rule-based")
        
        return rule_result
    except Exception as e:
        # Never throw; return empty/low confidence
        return ParseResponse(
            intents=["unknown"],
            consent=Consent(value="unknown", confidence=0.2),
            constraints=Constraints(confidence=0.2),
            availability=[],
            eligibility=EligibilityHints(confidence=0.2),
            followup=None
        )
