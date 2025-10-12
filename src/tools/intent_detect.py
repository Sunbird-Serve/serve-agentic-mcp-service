"""
Intent Detection Tool - Detect user intent to control conversation flow
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

# ------------- MODEL DEFINITIONS -------------

class IntentRequest(BaseModel):
    text: str
    context: Optional[str] = None  # "time_selection", "profile_setup", etc.

class IntentResponse(BaseModel):
    intent: str  # Primary intent
    confidence: float  # 0.0 - 1.0
    sub_intent: Optional[str] = None
    explanation: str
    suggested_action: str  # What agent should do next

# ------------- INTENT PATTERNS -------------

CONFIRMATION_PATTERNS = [
    "yes", "yeah", "yep", "sure", "ok", "okay", "fine", "correct", "right",
    "that works", "sounds good", "perfect", "great", "good", "confirmed",
    "book it", "let's do it", "proceed", "go ahead", "confirm"
]

REJECTION_PATTERNS = [
    "no", "nope", "nah", "not really", "doesn't work", "can't", "won't work",
    "not good", "not suitable", "not available", "busy"
]

REFINEMENT_PATTERNS = [
    "change", "different", "instead", "actually", "rather", "prefer",
    "better", "earlier", "later", "add", "remove", "shift", "modify"
]

QUESTION_PATTERNS = [
    "what", "when", "where", "who", "why", "how", "which", "can you",
    "could you", "would you", "?", "tell me", "show me"
]

GREETING_PATTERNS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "howdy"
]

HELP_PATTERNS = [
    "help", "how to", "what can you", "what do you", "options", "commands"
]

# ------------- INTENT DETECTION LOGIC -------------

def _detect_intent(text: str, context: Optional[str] = None) -> tuple[str, float, Optional[str], str, str]:
    """
    Detect user intent from text.
    
    Returns: (intent, confidence, sub_intent, explanation, suggested_action)
    
    Intents:
    - confirmation: User accepts/confirms
    - rejection: User rejects/declines
    - refinement: User wants to modify
    - question: User asking a question
    - greeting: User greeting
    - help: User needs help
    - time_request: User requesting time slots (context-specific)
    - unclear: Cannot determine
    """
    lower = text.lower().strip()
    
    # Very short responses
    if len(lower) <= 3:
        if lower in ["yes", "ok", "yep", "yea", "k"]:
            return (
                "confirmation",
                0.95,
                None,
                "Short affirmative response",
                "proceed_with_confirmed_action"
            )
        elif lower in ["no", "nah", "nope"]:
            return (
                "rejection",
                0.95,
                None,
                "Short negative response",
                "ask_for_alternative"
            )
    
    # Confirmation
    if any(pattern in lower for pattern in CONFIRMATION_PATTERNS):
        return (
            "confirmation",
            0.85,
            None,
            f"User confirmed with phrase in: {lower}",
            "proceed_with_confirmed_action"
        )
    
    # Rejection
    if any(pattern in lower for pattern in REJECTION_PATTERNS):
        has_refinement = any(pattern in lower for pattern in REFINEMENT_PATTERNS)
        
        if has_refinement:
            return (
                "refinement",
                0.9,
                "rejection_with_alternative",
                "User rejected but provided alternative",
                "refine_slots_with_new_request"
            )
        else:
            return (
                "rejection",
                0.8,
                None,
                "User declined without alternative",
                "ask_for_alternative"
            )
    
    # Refinement
    if any(pattern in lower for pattern in REFINEMENT_PATTERNS):
        return (
            "refinement",
            0.85,
            None,
            f"User wants to modify selection",
            "call_time_refine_slots"
        )
    
    # Question
    if any(pattern in lower for pattern in QUESTION_PATTERNS):
        return (
            "question",
            0.8,
            None,
            "User asking a question",
            "provide_information_or_help"
        )
    
    # Greeting
    if any(pattern in lower for pattern in GREETING_PATTERNS):
        return (
            "greeting",
            0.9,
            None,
            "User greeting",
            "respond_with_greeting_and_offer_help"
        )
    
    # Help
    if any(pattern in lower for pattern in HELP_PATTERNS):
        return (
            "help",
            0.85,
            None,
            "User needs help",
            "provide_help_or_options"
        )
    
    # Context-specific: Time selection
    if context == "time_selection":
        # Check if it looks like a time expression
        if any(word in lower for word in ["tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "afternoon", "evening", "night", "pm", "am", "o'clock"]):
            return (
                "time_request",
                0.75,
                None,
                "Looks like a time expression",
                "call_time_parse_or_refine"
            )
    
    # Unclear
    return (
        "unclear",
        0.3,
        None,
        "Could not determine intent",
        "ask_for_clarification"
    )

# ------------- MCP ENDPOINT -------------

@router.post("/intent.detect", response_model=IntentResponse)
async def detect_intent(req: IntentRequest):
    """
    MCP Tool: intent.detect
    
    Detects user intent to help control conversation flow and prevent loops.
    
    Primary use case: Determine if user is confirming a selection vs refining it.
    
    Examples:
    
    1. Confirmation:
       Input: "yes, that works"
       Output: {"intent": "confirmation", "suggested_action": "proceed_with_confirmed_action"}
    
    2. Refinement:
       Input: "no, make it friday instead"
       Output: {"intent": "refinement", "suggested_action": "call_time_refine_slots"}
    
    3. Rejection:
       Input: "no, that doesn't work"
       Output: {"intent": "rejection", "suggested_action": "ask_for_alternative"}
    
    4. Question:
       Input: "what other times do you have?"
       Output: {"intent": "question", "suggested_action": "provide_information_or_help"}
    
    Agent should use this to decide:
    - confirmation → proceed with booking
    - refinement → call time.refine_slots
    - rejection → ask for new times
    - unclear → ask for clarification
    """
    print(f"[intent.detect] Text: '{req.text}', Context: {req.context}")
    
    intent, confidence, sub_intent, explanation, suggested_action = _detect_intent(
        req.text,
        req.context
    )
    
    print(f"[intent.detect] Result: {intent} (confidence: {confidence}), Action: {suggested_action}")
    
    return IntentResponse(
        intent=intent,
        confidence=confidence,
        sub_intent=sub_intent,
        explanation=explanation,
        suggested_action=suggested_action
    )

