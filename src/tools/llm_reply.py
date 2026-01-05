"""
Reply Generation Tool - Hybrid approach (templates + LLM)
"""
from pydantic import BaseModel
from fastapi import APIRouter
from typing import Optional
from .llm_core import call_llm_for_text

router = APIRouter()

# ------------- MODEL DEFINITIONS -------------

class ReplyInput(BaseModel):
    purpose: str  # "ask_next_question" | "confirm" | "fallback" | "greeting" | "help"
    state: str
    profile_partial: dict = {}
    locale: str = "en-IN"
    use_llm: bool = True  # Toggle LLM enhancement
    user_name: Optional[str] = None  # For personalization

class ReplyOutput(BaseModel):
    text: str
    generation_method: str = "template"  # "template", "llm", or "hybrid"

# ------------- TEMPLATE-BASED REPLIES (FAST) -------------

def _template_reply(purpose: str, state: str, profile: dict, locale: str, user_name: Optional[str] = None) -> tuple[str, bool]:
    """
    Generate reply using templates.
    Returns: (reply_text, is_confident)
    - is_confident=True: Template is good enough, no LLM needed
    - is_confident=False: Template is generic, LLM should enhance
    """
    name_prefix = f"{user_name}, " if user_name else ""
    
    # Greeting
    if purpose == "greeting":
        if locale == "hi-IN":
            return f"नमस्ते {name_prefix}! मैं आपकी मदद कैसे कर सकता हूं?", True
        return f"Hello {name_prefix}! How can I help you today?", True
    
    # Help
    if purpose == "help":
        return "I can help you set up your teaching profile. Just tell me what subjects you teach, which grades, and when you're available!", True
    
    # Ask next question based on state
    if purpose == "ask_next_question":
        if state == "ASK_SUBJECTS":
            return "What subjects would you like to teach?", False  # Generic, can be improved by LLM
        
        elif state == "ASK_GRADES":
            subjects = ", ".join(profile.get("subjects", []))
            if subjects:
                return f"Great! {subjects} - which grade levels would you prefer?", True
            return "Which grade levels would you like to teach?", False
        
        elif state == "ASK_AVAILABILITY":
            return "When are you typically available to teach?", False
    
    # Confirm
    if purpose == "confirm":
        subjects = ", ".join(profile.get("subjects", [])) or "N/A"
        grades = ", ".join(profile.get("grades", [])) or "N/A"
        availability = profile.get("availability", "N/A")
        
        return f"Let me confirm:\n📚 Subjects: {subjects}\n🎓 Grades: {grades}\n📅 Availability: {availability}\n\nIs this correct?", True
    
    # Fallback
    if purpose == "fallback":
        return "I didn't quite understand that. Could you please rephrase?", False
    
    # Default
    return "Got it. Please continue.", False

# ------------- LLM ENHANCEMENT -------------

async def _llm_enhance(
    purpose: str,
    state: str,
    profile: dict,
    locale: str,
    template_reply: str,
    user_name: Optional[str] = None
) -> str:
    """
    Use LLM to make template reply more natural and personalized.
    """
    name_info = f"User's name: {user_name}" if user_name else "User's name: unknown"
    
    prompt = f"""You are a friendly teacher onboarding assistant. Make this reply more natural and engaging.

Context:
- Purpose: {purpose}
- Conversation state: {state}
- User profile so far: {profile}
- Locale: {locale}
- {name_info}

Template reply: "{template_reply}"

Instructions:
1. Keep the core message from the template
2. Make it more conversational and warm
3. Add encouraging phrases
4. Use appropriate language for {locale} (English for en-IN, Hindi for hi-IN)
5. Keep it concise (1-2 sentences)
6. Don't add emojis unless template has them

Enhanced reply:"""

    text, error = await call_llm_for_text(
        prompt=prompt,
        temperature=0.7,  # More creative for natural conversation
        max_tokens=150
    )
    
    if error:
        # LLM failed, return template
        return template_reply
    
    return text.strip()

# ------------- MCP ENDPOINT -------------

@router.post("/llm.generate_reply", response_model=ReplyOutput)
async def generate_reply(body: ReplyInput):
    """
    MCP Tool: llm.generate_reply (Hybrid)
    
    Generates conversational replies using a two-stage approach:
    1. Template-based reply (< 1ms) - structured, reliable
    2. LLM enhancement (optional) - natural, personalized
    
    Examples:
    
    Template only (fast, confident):
    Input: {purpose: "greeting"}
    Output: "Hello! How can I help you?" (method: "template")
    
    Hybrid (template + LLM):
    Input: {purpose: "ask_next_question", state: "ASK_SUBJECTS"}
    Template: "What subjects would you like to teach?"
    LLM enhances: "That's wonderful! What subjects are you passionate about teaching?"
    
    Set use_llm=false to skip LLM enhancement (faster, less personalized).
    """
    print(f"[llm.generate_reply] Purpose: {body.purpose}, State: {body.state}, Use LLM: {body.use_llm}")
    
    # Stage 1: Generate template-based reply
    template_text, is_confident = _template_reply(
        body.purpose,
        body.state,
        body.profile_partial,
        body.locale,
        body.user_name
    )
    
    print(f"[llm.generate_reply] Template: '{template_text}', Confident: {is_confident}")
    
    # If LLM disabled or template is confident, return template
    if not body.use_llm or is_confident:
        return ReplyOutput(
            text=template_text,
            generation_method="template"
        )
    
    # Stage 2: LLM enhancement for generic templates
    print("[llm.generate_reply] Template not confident, calling LLM for enhancement")
    enhanced_text = await _llm_enhance(
        body.purpose,
        body.state,
        body.profile_partial,
        body.locale,
        template_text,
        body.user_name
    )
    
    try:
        preview = enhanced_text.encode("cp1252", errors="ignore").decode("cp1252")
        print(f"[llm.generate_reply] LLM enhanced (ascii preview): '{preview}'")
    except Exception:
        print("[llm.generate_reply] LLM enhanced (preview unavailable due to encoding)")
    
    return ReplyOutput(
        text=enhanced_text,
        generation_method="hybrid"
    )
