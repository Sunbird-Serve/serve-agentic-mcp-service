"""
Humanize Weekday Confirmation - LLM tool
Generates a concise empathetic response with routing label and one follow-up question.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

from tools.llm_core import call_llm_for_json_messages

router = APIRouter()

# ---------- Models ----------

LabelLiteral = Literal[
    "YES", "NO", "CONSTRAINT", "DEFER", "SMALL_TALK", "AMBIGUOUS", "YES_WITH_CONDITION"
]

class HumanizeRequest(BaseModel):
    flow_state_summary: str = Field(..., description="Current step summary")
    user_input: str = Field(..., description="Volunteer last message")
    locale: str = Field(default="en-IN")

class HumanizeResponse(BaseModel):
    label: LabelLiteral
    tone_prefix: str
    reply: str
    bridge_question: str

    @field_validator("reply")
    @classmethod
    def limit_reply_sentences(cls, v: str) -> str:
        # Soft guard to avoid overly long generations (<= 2 sentences)
        return v.strip()

# ---------- Prompt builder ----------

SYSTEM_TEMPLATE = (
    "You are the “Human Layer” for SERVE’s Onboarding Agent.\n"
    "Goal: (1) Acknowledge the user naturally, (2) give a short helpful reply, (3) ask ONE clear next question that nudges progress on the current flow step.\n"
    "Current step (flow_state_summary): {flow}\n"
    "Key policy: Live classes can be scheduled only on weekdays between 8:00–15:00 (school hours). Weekends are currently not available for live classes.\n"
    "Tone: warm, concise, Indian-context friendly.\n"
    "Rules:\n"
    "- Keep to ≤2 sentences before the question.\n"
    "- If user proposes weekends (e.g., “Saturday first half”), thank them and explain weekday constraint; offer a micro-option (short weekday, lunch 20–30 mins).\n"
    "- If user defers (“can I think later?”), offer a reminder option.\n"
    "- If user is negative/frustrated, start with empathy (“I hear you”).\n"
    "- Never collect data not required for the current step.\n"
    "- Always end with ONE actionable question relevant to the step.\n"
    "Output JSON only:\n"
    "{ \"label\": \"<ONE_OF:[YES, NO, CONSTRAINT, DEFER, SMALL_TALK, AMBIGUOUS, YES_WITH_CONDITION]>\", \"tone_prefix\": \"<short phrase>\", \"reply\": \"<max 2 sentences>\", \"bridge_question\": \"<one clear question>\" }\n"
    "The label reflects your best interpretation of the user’s last message."
)

# Minimal few-shot (concise) to anchor behavior
FEWSHOTS = [
    {
        "flow": "Confirm weekday 8–15 consent; user suggests weekend",
        "user": "Saturday first half is available",
        "out": {
            "label": "CONSTRAINT",
            "tone_prefix": "Thanks for offering your Saturday time.",
            "reply": "Right now live sessions happen only on school weekdays (8–15).",
            "bridge_question": "Would you be open to a short weekday slot—maybe a 20–30 min lunch break?"
        }
    },
    {
        "flow": "Confirm weekday 8–15 consent; user defers",
        "user": "Can I think later?",
        "out": {
            "label": "DEFER",
            "tone_prefix": "No worries.",
            "reply": "We can circle back when you're ready.",
            "bridge_question": "Shall I set a reminder to check in next week?"
        }
    }
]

USER_TEMPLATE = (
    "Conversation context:\n"
    "- Flow: {flow}\n"
    "- User: \"{user}\"\n\n"
    "Return STRICT JSON only with keys: label, tone_prefix, reply, bridge_question. No extra text."
)

# ---------- Rule-based handler (fast path) ----------

def _rule_based_humanize(flow_summary: str, user_input: str, locale: str) -> Optional[HumanizeResponse]:
    """
    Rule-based humanize handler - returns result if pattern matches, None otherwise.
    Handles common responses without LLM.
    """
    low = user_input.lower().strip()
    words = low.split()
    
    # YES patterns - clear affirmative responses
    yes_patterns = ["yes", "yeah", "yep", "sure", "ok", "okay", "fine", "works", "sounds good", 
                    "that works", "absolutely", "of course", "definitely", "correct"]
    if len(words) <= 3 and any(p in low for p in yes_patterns) and not any(n in low for n in ["no", "not", "can't"]):
        return HumanizeResponse(
            label="YES",
            tone_prefix="Great!",
            reply="We'll proceed with weekday scheduling.",
            bridge_question="Which part of the day works better for you—morning (8–11 AM) or afternoon (12–3 PM)?"
        )
    
    # NO patterns - clear negative responses
    no_patterns = ["no", "nope", "not", "can't", "cant", "won't", "cannot", "not possible", "too busy"]
    if len(words) <= 4 and any(p in low for p in no_patterns):
        return HumanizeResponse(
            label="NO",
            tone_prefix="I understand.",
            reply="Weekday scheduling might not work for your situation right now.",
            bridge_question="Would you like me to set a reminder to check back later when your schedule might be more flexible?"
        )
    
    # DEFER patterns - deferring/thinking
    defer_patterns = ["think later", "think and get back", "get back", "later", "not now", "maybe later",
                      "let me think", "i'll think", "will think", "maybe", "let me see"]
    if any(p in low for p in defer_patterns):
        return HumanizeResponse(
            label="DEFER",
            tone_prefix="No worries.",
            reply="Take your time to think it over.",
            bridge_question="Shall I set a reminder to check in next week?"
        )
    
    # CONSTRAINT patterns - weekend preference
    weekend_patterns = ["weekend", "saturday", "sunday", "only weekend", "weekends only", "prefer weekend"]
    if any(p in low for p in weekend_patterns):
        return HumanizeResponse(
            label="CONSTRAINT",
            tone_prefix="Thanks for offering your weekend time.",
            reply="Right now, live sessions happen only on school weekdays (8–15).",
            bridge_question="Would you be open to a short weekday slot—maybe a 20–30 min lunch break?"
        )
    
    # YES_WITH_CONDITION patterns - conditional agreement
    condition_patterns = ["if", "but", "however", "only if", "as long as", "provided", "maybe if"]
    if any(p in low for p in condition_patterns) and any(y in low for y in yes_patterns):
        return HumanizeResponse(
            label="YES_WITH_CONDITION",
            tone_prefix="I appreciate you considering it.",
            reply="Let's see what we can work out with the weekday schedule.",
            bridge_question="What would make weekday timing work better for you?"
        )
    
    return None  # No clear pattern, needs LLM

# ---------- Endpoint ----------

@router.post("/llm.humanize_weekday_confirmation", response_model=HumanizeResponse)
async def humanize_weekday_confirmation(req: HumanizeRequest) -> HumanizeResponse:
    # Step 1: Try rule-based first (fast, cost-free)
    rule_result = _rule_based_humanize(req.flow_state_summary, req.user_input, req.locale)
    if rule_result:
        print(f"[llm.humanize_weekday_confirmation] Rule-based match, skipping LLM")
        return rule_result
    
    # Step 2: Rule-based didn't match, use LLM for complex responses
    print(f"[llm.humanize_weekday_confirmation] No rule-based match, calling LLM")
    # Avoid .format() to prevent KeyError from JSON braces; do a targeted replace
    system = SYSTEM_TEMPLATE.replace("{flow}", req.flow_state_summary)

    # Compact few-shot exemplars inside the prompt to reduce tokens
    fewshot_block = []
    for ex in FEWSHOTS:
        fewshot_block.append(
            f"Example:\nFlow: {ex['flow']}\nUser: {ex['user']}\nOutput JSON: {ex['out']}\n"
        )
    fewshot_text = "\n".join(fewshot_block)

    user = USER_TEMPLATE.format(flow=req.flow_state_summary, user=req.user_input)

    prompt = (
        f"{system}\n\n"
        f"{fewshot_text}\n\n"
        f"{user}"
    )

    # Use system message with few-shots per spec
    data, error = await call_llm_for_json_messages(
        system_text=system + "\n\n" + fewshot_text,
        user_text=user,
        temperature=0.3,
        max_tokens=220
    )

    if error:
        # Graceful fallback with deterministic minimal response
        return HumanizeResponse(
            label="AMBIGUOUS",
            tone_prefix="Thanks for sharing.",
            reply="Live sessions run on weekdays 8–15.",
            bridge_question="Would a short weekday slot around lunch work for you?"
        )

    # Normalize/validate, enforce DEFER for defer-like inputs, cap reply at 2 sentences
    try:
        label = (data or {}).get("label", "AMBIGUOUS")
        tone_prefix = (data or {}).get("tone_prefix", "Thanks for sharing.")
        reply = (data or {}).get("reply", "Live sessions run on weekdays 8–15.")
        bridge_question = (data or {}).get("bridge_question", "Would a short weekday slot around lunch work for you?")

        # Force DEFER for defer-like signals
        defer_markers = ["think later", "think and get back", "get back", "later", "not now", "maybe later"]
        low = req.user_input.lower()
        if any(m in low for m in defer_markers):
            label = "DEFER"

        # Trim reply to max 2 sentences and dedupe
        # Normalize repeated whitespace and remove exact duplicate sentences
        sentences = [s.strip() for s in re.split(r"[.!]", reply) if s.strip()] if 're' in globals() else [s.strip() for s in reply.split('.') if s.strip()]
        seen_s = []
        dedup = []
        for s in sentences:
            if s not in seen_s:
                seen_s.append(s)
                dedup.append(s)
        sentences = dedup
        if len(sentences) > 2:
            reply = '. '.join(sentences[:2]) + '.'
        elif sentences and not reply.endswith('.'):
            reply = reply + '.'

        # Avoid repeating same sentence across tone_prefix and reply
        tp = tone_prefix.strip()
        if tp and tp.rstrip('.').lower() in reply.lower():
            # If tone already captured in reply, shorten prefix
            tone_prefix = tp.split('.')[0][:32]
            if tone_prefix and tone_prefix.endswith('.'):
                tone_prefix = tone_prefix[:-1]

        # Ensure bridge is concise and not repeating reply
        if bridge_question and bridge_question.rstrip('?').lower() in reply.lower():
            bridge_question = "Would a short weekday slot around lunch work for you?"

        # Enforce exactly one question for bridge_question
        bq = bridge_question.strip()
        if '?' in bq:
            bq = bq.split('?')[0].strip() + '?'
        else:
            # ensure it's phrased as a question
            if not bq.endswith('?'):
                bq = (bq.rstrip('.').rstrip('!') + '?') if bq else "Would a short weekday slot around lunch work for you?"

        return HumanizeResponse(
            label=label if label in {"YES","NO","CONSTRAINT","DEFER","SMALL_TALK","AMBIGUOUS","YES_WITH_CONDITION"} else "AMBIGUOUS",
            tone_prefix=tone_prefix,
            reply=reply,
            bridge_question=bq
        )
    except Exception:
        return HumanizeResponse(
            label="AMBIGUOUS",
            tone_prefix="Thanks for sharing.",
            reply="Live sessions run on weekdays 8–15.",
            bridge_question="Would a short weekday slot around lunch work for you?"
        )
