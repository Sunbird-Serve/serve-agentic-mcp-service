"""
DEPRECATED: This tool orchestrates conversation flow, which violates MCP architecture.

The agent (MCP client) should own conversation orchestration.
The server should only expose discrete tools (business logic, parsing, capabilities).

This tool is kept for backward compatibility only.
New implementations should use discrete tools:
- onboarding.parse_message - Parse user input
- eligibility.check - Check eligibility (business logic)
- consent.record - Record consent
- llm.call - Generate messages when needed
- slots.propose, slot.book, etc.

Agent should implement state machine and orchestrate tool calls.
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import httpx
import json

router = APIRouter()

class Session(BaseModel):
    state: str
    profile: Dict[str, Any] = {}
    ts: Optional[Any] = None
    conversation_history: Optional[List[Dict[str, str]]] = None  # [{role: "user"|"assistant", content: "..."}]

class NextRequest(BaseModel):
    session: Session
    user_text: str
    locale: str = "en-IN"

class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any]

class NextResponse(BaseModel):
    next_state: str
    message: str
    quick_replies: Optional[List[str]] = None
    calls: Optional[List[ToolCall]] = None
    updates: Optional[Dict[str, Any]] = None

MCP_BASE = "http://127.0.0.1:9000/mcp"

async def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{MCP_BASE}{path}", json=payload)
        r.raise_for_status()
        return r.json()

async def _parse(text: str, locale: str) -> Dict[str, Any]:
    return await _post("/onboarding.parse_message", {"text": text, "locale": locale})

async def _humanize(flow: str, user: str, locale: str) -> Dict[str, Any]:
    return await _post("/llm.humanize_weekday_confirmation", {"flow_state_summary": flow, "user_input": user, "locale": locale})

async def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 300, temperature: float = 0.7) -> Optional[str]:
    """Call llm.call with messages array"""
    try:
        response = await _post("/llm.call", {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        })
        return response.get("content") or response.get("message") or response.get("text")
    except Exception as e:
        print(f"[onboarding.next] LLM call error: {e}")
        return None

def _get_state_context(state: str, profile: Dict[str, Any]) -> str:
    """Build context string for current state"""
    state_upper = state.upper()
    
    context_parts = []
    
    if state_upper in ("GREET", "WELCOME"):
        context_parts.append("STEP: Welcome message, introduce SERVE")
        context_parts.append("ACTION: Greet volunteer, explain SERVE briefly, ask if ready to start")
    
    elif state_upper == "CONSENT_NO_PAY":
        context_parts.append("STEP: Explain volunteer role (no payment), get consent")
        context_parts.append("ACTION: Be transparent about no payment, frame positively (making impact), ask if okay")
        context_parts.append("KEY: This is a critical consent step - must be clear about volunteer nature")
    
    elif state_upper == "ELIGIBILITY_PART1":
        context_parts.append("STEP: Check age (18+) and device (smartphone/laptop with internet)")
        context_parts.append("ACTION: Ask both questions together, confirm both are met")
        age = profile.get("eligibility", {}).get("q2_age")
        device = profile.get("eligibility", {}).get("q3_device")
        collected = []
        if age is not None: collected.append("age")
        if device is not None: collected.append("device")
        if collected:
            context_parts.append(f"COLLECTED: {', '.join(collected)}")
        else:
            context_parts.append("COLLECTED: none yet")
    
    elif state_upper == "ELIGIBILITY_PART2":
        context_parts.append("STEP: Check commitment (2hrs/week for at least 3 months)")
        context_parts.append("ACTION: Explain importance of consistency, ask for commitment")
        commitment = profile.get("eligibility", {}).get("q1_commitment")
        if commitment is not None:
            context_parts.append("COLLECTED: commitment")
        else:
            context_parts.append("COLLECTED: not yet")
    
    elif state_upper == "CLASS_CONSTRAINTS":
        context_parts.append("STEP: Confirming weekday availability (8 AM-3 PM, Mon-Fri only)")
        context_parts.append("POLICY: Classes run only on weekdays 8:00-15:00 (school hours). Weekends not available.")
    
    elif state_upper == "TIME_PREF":
        context_parts.append("STEP: Selecting time band preference (8-11 AM or 12-3 PM)")
        time_band = profile.get("timeBand")
        if time_band:
            context_parts.append(f"CURRENT PREFERENCE: {time_band}")
    
    elif state_upper == "SLOTING":
        context_parts.append("STEP: Proposing specific 30-minute slots within chosen time band")
        slots = profile.get("slots", [])
        if slots:
            context_parts.append(f"AVAILABLE SLOTS: {len(slots)} options shown")
    
    elif state_upper == "SLOT_SELECT":
        context_parts.append("STEP: User selecting which slot to hold")
    
    elif state_upper == "HOLD_CONFIRM":
        context_parts.append("STEP: Confirming slot booking")
    
    elif state_upper == "WRAP":
        context_parts.append("STEP: Final wrap-up, optional reminder setup")
    
    return "\n".join(context_parts)

def _build_system_prompt(state: str, profile: Dict[str, Any], locale: str) -> str:
    """Build system prompt for LLM with state-aware context"""
    state_context = _get_state_context(state, profile)
    volunteer_name = profile.get("name") or ""
    
    return f"""You are a warm, friendly onboarding assistant for SERVE, a volunteer teaching platform in India.

YOUR ROLE:
- Guide volunteers through onboarding with empathy and clarity
- Make conversations natural and conversational, not robotic
- Remember context from previous messages
- Be concise (2-3 sentences max per response)
- Use Indian English, be culturally aware

CURRENT SITUATION:
{state_context}

POLICY CONSTRAINTS:
- Live classes run ONLY on weekdays (Mon-Fri), 8:00 AM - 3:00 PM (school hours)
- Weekends are NOT available for live classes
- Each class session is typically 30 minutes
- This is a volunteer role (no pay)
- Minimum commitment: 2 hours per week

VOLUNTEER PROFILE:
- Name: {volunteer_name if volunteer_name else "Not yet collected"}
- Subjects: {', '.join(profile.get('subjects', [])) or 'Not yet collected'}
- Grades: {profile.get('grades') or 'Not yet collected'}
- Language: {profile.get('language') or 'Not yet collected'}
- Eligibility: {profile.get('eligibility', {})}

CONVERSATION FLOW STATES:
1. GREET → Welcome message, introduce SERVE
2. CONSENT_NO_PAY → Explain volunteer role (no payment), get consent
3. ELIGIBILITY_PART1 → Check age (18+) and device (smartphone/laptop with internet)
4. ELIGIBILITY_PART2 → Check commitment (2hrs/week for at least 3 months)
5. CLASS_CONSTRAINTS → Confirm weekday 8-15 availability (if weekend preference, explain constraint with empathy)
6. TIME_PREF → Choose time band (8-11 AM or 12-3 PM)
7. SLOTING → Show 2-3 specific 30-minute slot options
8. SLOT_SELECT → User picks a slot
9. HOLD_CONFIRM → Confirm booking
10. WRAP → Final wrap-up, reminder setup

YOUR TASK:
For each user message, respond naturally in JSON format:
{{
  "message": "Your natural, warm response (2-3 sentences max)",
  "next_state": "Next state name (GREET, CONSENT, ELIGIBILITY_PART1, CLASS_CONSTRAINTS, TIME_PREF, SLOTING, SLOT_SELECT, HOLD_CONFIRM, WRAP)",
  "tool_calls": [
    {{"tool": "consent.record", "args": {{"volunteerId": "...", "consentGiven": true}}}},
    {{"tool": "eligibility.check", "args": {{"ageYears": 18, "hasDevice": true, "weeklyCommitmentHours": 2.0}}}},
    {{"tool": "preferences.save", "args": {{"volunteerId": "...", "timeBand": "8-11"}}}},
    {{"tool": "slots.propose", "args": {{"volunteerId": "...", "timeBand": "8-11", "limit": 2}}}},
    {{"tool": "slot.hold", "args": {{"volunteerId": "...", "slotId": "..."}}}},
    {{"tool": "slot.book", "args": {{"volunteerId": "...", "slotId": "..."}}}},
    {{"tool": "reminder.create", "args": {{"volunteerId": "...", "when_ISO": "...", "reason": "..."}}}}
  ],
  "updates": {{
    "profile": {{"key": "value"}}  // Optional profile updates
  }},
  "quick_replies": ["Option 1", "Option 2"]  // Optional quick reply buttons
}}

RULES:
- ALWAYS return ONLY valid JSON - no markdown code blocks, no explanations, no extra text
- Keep message warm, natural, conversational (2-3 sentences max)
- Progress through states logically based on conversation flow
- Use tool_calls to execute actions (consent.record, eligibility.check, etc.) when needed
- If user asks questions or expresses concerns, address them empathetically but stay on track
- If user wants to restart, set next_state to "GREET"
- If user declines or says no, gracefully wrap up with next_state "WRAP"
- Never mention weekends as an option - explain the constraint with empathy
- Remember conversation history - reference previous messages when relevant

CRITICAL: Return ONLY the JSON object, nothing else. Example format:
{{"message": "Hello!", "next_state": "CONSENT", "tool_calls": [], "quick_replies": ["Yes", "No"]}}
"""

def _is_simple_response(text: str) -> bool:
    """Check if response is simple enough for rule-based handling (skip LLM)"""
    low = text.lower().strip()
    words = low.split()
    
    # Simple responses: 1-4 words, clear yes/no/ok patterns
    if len(words) > 4:
        return False  # Too complex, needs LLM
    
    # Simple yes/affirmative patterns (including "go ahead", "let's go", etc.)
    yes_patterns = [
        "yes", "yeah", "yep", "ok", "okay", "sure", "fine", "alright", "correct", 
        "go ahead", "lets go", "let's go", "proceed", "continue", "ready", 
        "sounds good", "works for me", "that's fine", "perfect"
    ]
    # Simple no patterns
    no_patterns = ["no", "nope", "not", "can't", "cant", "won't", "dont want"]
    # Simple defer patterns
    defer_patterns = ["later", "think", "maybe", "not now", "not yet"]
    
    # Check if it's a simple clear response
    is_simple = (
        any(p in low for p in yes_patterns) and not any(n in low for n in no_patterns)
    ) or any(p in low for p in no_patterns) or any(p in low for p in defer_patterns)
    
    return is_simple and len(words) <= 4

@router.post("/onboarding.next", response_model=NextResponse)
async def onboarding_next(req: NextRequest) -> NextResponse:
    s = req.session
    text = (req.user_text or "").strip()
    state = s.state or "GREET"
    
    # Handle empty user_text (initial greeting or kickoff)
    if not text:
        print(f"[onboarding.next] Empty user_text, returning initial greeting for state: {state}")
        # Use fallback for initial state handling
        result = await _fallback_response(state, "", s.profile, req.locale)
        # Initialize conversation history with assistant greeting
        history = s.conversation_history or []
        history.append({"role": "assistant", "content": result.message})
        if result.updates is None:
            result.updates = {}
        result.updates["conversation_history"] = history[-20:]
        return result
    
    # Initialize conversation history if not present
    history = s.conversation_history or []
    
    # Add current user message to history
    history.append({"role": "user", "content": text})
    
    # HYBRID APPROACH: Fast path for simple responses (skip LLM, reduce cost)
    if _is_simple_response(text):
        print(f"[onboarding.next] Simple response detected: '{text}', using rule-based fallback (no LLM)")
        result = await _fallback_response(state, text, s.profile, req.locale)
        # Update conversation history with assistant response
        history.append({"role": "assistant", "content": result.message})
        # Include updated history in updates
        if result.updates is None:
            result.updates = {}
        result.updates["conversation_history"] = history[-20:]
        return result
    
    # Complex response - use LLM for natural conversation
    print(f"[onboarding.next] Complex response: '{text}', calling LLM")
    
    # Build system prompt with current state and profile
    system_prompt = _build_system_prompt(state, s.profile, req.locale)
    
    # Build messages array for LLM
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation history (last 10 messages to keep context manageable)
    # Filter out empty messages to avoid API errors
    for msg in history[-10:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content") and msg.get("content").strip():
            messages.append({"role": msg["role"], "content": msg["content"].strip()})
    
    # Add current user message (already validated as non-empty)
    messages.append({"role": "user", "content": text})
    
    print(f"[onboarding.next] State: {state}, Calling LLM with {len(messages)} messages")
    
    # Call LLM for natural conversation response
    llm_response = await _call_llm(
        messages=messages,
        max_tokens=400,  # More tokens for structured JSON response
        temperature=0.7  # Balanced creativity
    )
    
    # Parse LLM response (expecting JSON)
    if llm_response:
        try:
            # Extract JSON from response (might have markdown code blocks)
            json_text = llm_response.strip()
            if json_text.startswith("```"):
                # Remove markdown code blocks
                json_text = json_text.split("```")[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:]
                json_text = json_text.strip()
            elif json_text.startswith("`"):
                json_text = json_text.strip("`")
            
            # Parse JSON
            parsed = json.loads(json_text)
            
            # Extract fields
            message = parsed.get("message", "")
            next_state = parsed.get("next_state", state)
            tool_calls_data = parsed.get("tool_calls", [])
            updates = parsed.get("updates", {})
            quick_replies = parsed.get("quick_replies")
            
            # Convert tool_calls to ToolCall objects
            calls = []
            for tc in tool_calls_data:
                if isinstance(tc, dict) and "tool" in tc and "args" in tc:
                    calls.append(ToolCall(tool=tc["tool"], args=tc["args"]))
            
            # Update conversation history with assistant response
            history.append({"role": "assistant", "content": message})
            
            # Fallback: If message is empty or invalid, use fallback logic
            if not message or len(message.strip()) < 5:
                print(f"[onboarding.next] LLM response too short, using fallback")
                return await _fallback_response(state, text, s.profile, req.locale)
            
            print(f"[onboarding.next] LLM generated response: '{message[:50]}...', next_state: {next_state}")
            
            # Include updated conversation history in updates
            if updates is None:
                updates = {}
            updates["conversation_history"] = history[-20:]  # Keep last 20 messages
            
            return NextResponse(
                next_state=next_state,
                message=message,
                quick_replies=quick_replies,
                calls=calls if calls else None,
                updates=updates
            )
            
        except json.JSONDecodeError as e:
            print(f"[onboarding.next] Failed to parse LLM JSON response: {e}")
            print(f"[onboarding.next] Raw response: {llm_response[:200]}")
            # Fallback to rule-based
            return await _fallback_response(state, text, s.profile, req.locale)
    
    # LLM call failed, use fallback
    print(f"[onboarding.next] LLM call failed, using fallback logic")
    return await _fallback_response(state, text, s.profile, req.locale)

async def _fallback_response(state: str, text: str, profile: Dict[str, Any], locale: str) -> NextResponse:
    """Fallback rule-based responses when LLM is unavailable"""
    state_upper = state.upper()
    low = text.lower().strip() if text else ""
    
    # Default GREET or WELCOME (initial state)
    if not state or state_upper in ("GREET", "WELCOME"):
        return NextResponse(
            next_state="CONSENT_NO_PAY",
            message=("Hi! I'm here to help you get started with teaching at SERVE. "
                     "Just a quick intro—we connect volunteers with students for live online classes. "
                     "Ready to get started?"),
            quick_replies=["Yes", "No", "Tell me more"]
        )
    
    # CONSENT_NO_PAY - Explain volunteer role (no payment), get consent
    if state_upper == "CONSENT_NO_PAY":
        parsed = await _parse(text, locale)
        consent = parsed.get("consent", {})
        val = (consent.get("value") or "unknown").lower()
        if val == "yes":
            calls = []
            volunteer_id = profile.get("volunteerId") or profile.get("id") or "vol-unknown"
            calls.append(ToolCall(tool="consent.record", args={"volunteerId": volunteer_id, "consentGiven": True}))
            return NextResponse(
                next_state="ELIGIBILITY_PART1",
                message=("Perfect! Before we dive in, let me quickly check: "
                         "Are you 18 or older, and do you have a smartphone or laptop with internet?"),
                quick_replies=["Yes, both", "I have a question", "Tell me more"],
                calls=calls
            )
        if val == "no":
            return NextResponse(
                next_state="WRAP",
                message=("Thanks for considering it! If you'd like, I can remind you later."),
                quick_replies=["Set a reminder", "No thanks"]
            )
        # If asking for more info or unclear
        if "tell me more" in low or "more" in low or "info" in low:
            return NextResponse(
                next_state="CONSENT_NO_PAY",
                message=("Great question! This is a fully volunteer role, which means there's no payment involved. "
                         "It's all about making a meaningful impact in students' lives through live online classes. "
                         "Are you okay with that?"),
                quick_replies=["Yes, that's fine", "I need to think", "No"]
            )
        return NextResponse(
            next_state="CONSENT_NO_PAY",
            message=("Just want to be upfront—this is a fully volunteer role (no payment). "
                     "It's about making a meaningful impact in students' lives. Are you okay with that?"),
            quick_replies=["Yes, that's fine", "I need to think", "Tell me more"]
        )

    # ELIGIBILITY_PART1 - Check age (18+) and device (smartphone/laptop with internet)
    if state_upper == "ELIGIBILITY_PART1":
        parsed = await _parse(text, locale)
        elig = parsed.get("eligibility", {})
        age_ok = elig.get("age_ok")
        has_device = elig.get("has_device") if elig.get("has_device") is not None else elig.get("device_ok")
        
        # If both confirmed
        if age_ok is True and has_device is True:
            return NextResponse(
                next_state="ELIGIBILITY_PART2",
                message=("Great! Last thing—can you commit about 2 hours per week "
                         "for at least 3 months? We find that consistency really helps the students."),
                quick_replies=["Yes, I can", "Tell me more", "I'm not sure"]
            )
        
        # If one or both missing, clarify
        if age_ok is None or has_device is None:
            missing = []
            if age_ok is None: missing.append("Are you 18 or older?")
            if has_device is None: missing.append("Do you have a smartphone or laptop with internet?")
            return NextResponse(
                next_state="ELIGIBILITY_PART1",
                message=("Let me check: " + " ".join(missing)),
                quick_replies=["Yes, both", "I have a question", "Tell me more"]
            )
        
        # If either is False, not eligible
        if age_ok is False or has_device is False:
            return NextResponse(
                next_state="WRAP",
                message=("Appreciate your interest! You may be better suited for non-teaching roles. "
                         "Shall I set a reminder if things change?"),
                quick_replies=["Set a reminder", "No thanks"]
            )
        
        # Default: ask again
        return NextResponse(
            next_state="ELIGIBILITY_PART1",
            message=("Are you 18 or older, and do you have a smartphone or laptop with internet?"),
            quick_replies=["Yes, both", "I have a question", "Tell me more"]
        )
    
    # ELIGIBILITY_PART2 - Check commitment (2hrs/week for at least 3 months)
    if state_upper == "ELIGIBILITY_PART2":
        parsed = await _parse(text, locale)
        elig = parsed.get("eligibility", {})
        weekly = elig.get("weekly_commitment_hours")
        commitment_hint = elig.get("commitment_hint", "")
        
        # Check if user confirmed commitment (yes patterns)
        low = text.lower().strip()
        if any(p in low for p in ["yes", "yeah", "yep", "sure", "ok", "okay", "can", "will", "absolutely", "definitely"]):
            # Extract hours if mentioned, default to 2.0
            if weekly is None:
                weekly = 2.0
            # Assume 3+ months if commitment confirmed
            calls: List[ToolCall] = []
            volunteer_id = profile.get("volunteerId") or profile.get("id") or "vol-unknown"
            check_result = await _post("/eligibility.check", {
                "ageYears": 18, 
                "hasDevice": True, 
                "weeklyCommitmentHours": float(weekly)
            })
            if check_result.get("eligible"):
                return NextResponse(
                    next_state="CLASS_CONSTRAINTS",
                    message=("Perfect! Thanks for that. Live school sessions run Mon–Fri, between 8 AM and 3 PM. "
                             "Which part of the day works better for you?"),
                    quick_replies=["8–11 AM", "12–3 PM", "Only weekends", "I'll think later"],
                    calls=calls
                )
            else:
                return NextResponse(
                    next_state="WRAP",
                    message=("Appreciate your interest! You may be better suited for non-teaching roles. "
                             "Shall I set a reminder if things change?"),
                    quick_replies=["Set a reminder", "No thanks"]
                )
        
        # If declined or unclear
        if any(p in low for p in ["no", "can't", "cant", "won't", "not sure", "dont know"]):
            return NextResponse(
                next_state="WRAP",
                message=("Thanks for considering it! If you'd like, I can remind you later when you're ready."),
                quick_replies=["Set a reminder", "No thanks"]
            )
        
        # Default: clarify commitment
        return NextResponse(
            next_state="ELIGIBILITY_PART2",
            message=("Just to confirm—can you commit about 2 hours per week for at least 3 months? "
                     "This consistency helps students a lot!"),
            quick_replies=["Yes, I can", "Tell me more", "I'm not sure"]
        )

    # CLASS_CONSTRAINTS -> TIME_PREF
    if state_upper == "CLASS_CONSTRAINTS":
        parsed = await _parse(text, locale)
        constraints = parsed.get("constraints", {})
        # Check for weekend-only preference
        if constraints.get("weekend_only") and not constraints.get("weekday_ok"):
            try:
                human = await _humanize("Confirm weekday 8–15 consent; user suggests weekend", text, locale)
                tone = human.get('tone_prefix', '').strip()
                reply = human.get('reply', '').strip()
                bridge = human.get('bridge_question', '').strip()
                parts = [p for p in [tone, reply, bridge] if p]
                msg = ' '.join(parts) if parts else "I understand you prefer weekends. Our classes run Mon–Fri, 8 AM–3 PM. Could a weekday slot work?"
            except Exception:
                msg = "I understand you prefer weekends. Our classes run Mon–Fri, 8 AM–3 PM. Could a weekday slot work?"
            return NextResponse(
                next_state="CLASS_CONSTRAINTS",
                message=msg,
            )
        # If weekday_ok is true (affirmative response), proceed to time band selection
        if constraints.get("weekday_ok") is True:
            return NextResponse(
                next_state="TIME_PREF",
                message="Which part of the day works better for you?",
                quick_replies=["8–11 AM", "12–3 PM"]
            )
        # Accept time band selection
        low = text.lower()
        if "8" in low and ("11" in low or "8–11" in low or "8-11" in low):
            vid = profile.get("volunteerId") or "vol-unknown"
            return NextResponse(
                next_state="TIME_PREF",
                message="Noted 8–11 AM. Shall I propose a couple of 30‑min options?",
                quick_replies=["Yes", "Show 12–3 instead"],
                calls=[ToolCall(tool="preferences.save", args={"volunteerId": vid, "timeBand": "8-11"})]
            )
        if "12" in low or "3 pm" in low or "12–3" in low or "12-3" in low:
            vid = profile.get("volunteerId") or "vol-unknown"
            return NextResponse(
                next_state="TIME_PREF",
                message="Noted 12–3 PM. Shall I propose a couple of 30‑min options?",
                quick_replies=["Yes", "Show 8–11 instead"],
                calls=[ToolCall(tool="preferences.save", args={"volunteerId": vid, "timeBand": "12-15"})]
            )
        # Clarify
        return NextResponse(
            next_state="CLASS_CONSTRAINTS",
            message="Which part works better for you?",
            quick_replies=["8–11 AM", "12–3 PM", "Only weekends", "I’ll think later"]
        )

    # TIME_PREF -> SLOTING
    if state_upper == "TIME_PREF":
        low = text.lower()
        vid = profile.get("volunteerId") or "vol-unknown"
        band = "8-11" if "8" in low else ("12-15" if "12" in low or "3" in low else profile.get("timeBand", "12-15"))
        calls = [ToolCall(tool="slots.propose", args={"volunteerId": vid, "timeBand": band, "limit": 2})]
        return NextResponse(
            next_state="SLOTING",
            message="Here are two quick options that fit your timing:",
            calls=calls
        )

    # SLOTING: client should show returned slots; next expects selection text
    if state_upper == "SLOTING":
        # Defer actual hold to client after reading last proposed slots; here we prompt confirm-style
        return NextResponse(
            next_state="SLOT_SELECT",
            message="Which one shall I hold for you?",
            quick_replies=["First option", "Second option", "Show others"]
        )

    if state_upper == "SLOT_SELECT":
        choice = text.lower()
        pick = 0
        if "second" in choice or "2" in choice:
            pick = 1
        return NextResponse(
            next_state="HOLD_CONFIRM",
            message="Holding your selected slot for 2 minutes. Shall I confirm this slot?",
            quick_replies=["Confirm", "Pick the other one"]
        )

    if state_upper == "HOLD_CONFIRM":
        if "confirm" in text.lower():
            return NextResponse(
                next_state="WRAP",
                message=("Done! Your class is booked. I can remind you a day before—shall I set a reminder?"),
                quick_replies=["Yes", "No"],
            )
        return NextResponse(
            next_state="SLOTING",
            message="No problem—let’s pick another option.",
            quick_replies=["Show others", "8–11 AM", "12–3 PM"]
        )

    # CONSENT_NO_PAY is already handled above in the main CONSENT_NO_PAY section
    # Keeping this for backward compatibility if needed
    
    # Handle restart intent
    if "restart" in text.lower() or "start over" in text.lower() or "begin again" in text.lower():
        return NextResponse(
            next_state="GREET",
            message=("Let's start fresh! Hi! I'm your SERVE onboarding assistant. "
                     "To begin, may I record your consent to proceed?"),
            quick_replies=["Yes", "No", "What info do you store?"]
        )
    
    # WRAP state - handle responses
    if state_upper == "WRAP":
        low = text.lower()
        if "yes" in low or "reminder" in low or "set reminder" in low:
            # User wants reminder or help
            return NextResponse(
                next_state="WRAP",
                message=("I'll set that up for you. Anything else?"),
                quick_replies=["No thanks", "Help"]
            )
        if "no" in low or "no thanks" in low or "nothing" in low:
            return NextResponse(
                next_state="WRAP",
                message=("Sounds good! Feel free to reach out if you need anything. Good luck with your teaching!")
            )
        if "help" in low or "support" in low:
            return NextResponse(
                next_state="WRAP",
                message=("I'm here to help with onboarding. You can ask me about the process, requirements, or restart anytime.")
            )
        # Default for WRAP
        return NextResponse(
            next_state="WRAP",
            message=("You're all set. Need anything else?"),
            quick_replies=["Set a reminder", "No thanks", "Help"]
        )
    
    # Unknown state - default to WRAP
    return NextResponse(
        next_state="WRAP",
        message="You're all set. Need anything else?",
        quick_replies=["Set a reminder", "No thanks", "Help"]
    )
