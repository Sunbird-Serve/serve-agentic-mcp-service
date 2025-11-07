"""
LLM-powered response classification for eligibility questions.

This tool uses Claude to intelligently classify user responses to eligibility questions
with natural language understanding, reducing false negatives and improving UX.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional, List, Any
from datetime import datetime, timedelta
import json
import re

from tools.llm_core import call_llm_for_json

router = APIRouter()

# ------------- REQUEST/RESPONSE MODELS -------------

class ClassifyRequest(BaseModel):
    """Request to classify a user response"""
    question_type: Literal["consent", "age", "device", "commitment", "language_comfort"] = Field(
        ...,
        description="Type of eligibility question being asked"
    )
    user_input: str = Field(
        ...,
        description="User's response text"
    )
    context: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional context (question_text, locale, etc.)"
    )

class ClassifyResponse(BaseModel):
    """Classification result"""
    classification: Literal["YES", "NO", "UNCLEAR"] = Field(
        ...,
        description="Classification result: YES (passes), NO (fails), UNCLEAR (needs clarification)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    extracted_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted information per type"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of the classification"
    )

# ------------- PROMPT TEMPLATES -------------

def _get_age_prompt(question_text: str, user_input: str) -> str:
    """Generate prompt for age question classifier"""
    return f"""You are analyzing a user's response to an eligibility question about age.

Context:
Question: {question_text}
User Response: "{user_input}"

Task: Classify whether the user confirms they are 18 years or older.

Classification Rules:
- YES: User confirms being 18+ 
  Examples: "yes", "I am 19", "of course", "yes I am", "I'm 20"
- NO: User indicates they are under 18
  Examples: "no", "I'm 17", "not yet", "only 16", "I'm 15"
- UNCLEAR: Ambiguous response that needs clarification
  Examples: "maybe", "I think so", "not sure", "depends"

Return your analysis as JSON:
{{
  "classification": "YES" | "NO" | "UNCLEAR",
  "confidence": 0.0-1.0,
  "extracted_info": {{"is_18_plus": true/false, "age_number": 19|null}},
  "reasoning": "Brief explanation"
}}

Return ONLY the JSON object. No markdown, no explanation."""

def _get_device_prompt(question_text: str, user_input: str) -> str:
    """Generate prompt for device question classifier"""
    return f"""You are analyzing a user's response to an eligibility question about devices.

Context:
Question: {question_text}
User Response: "{user_input}"

Task: Classify whether the user has a suitable device for online teaching.

Suitable Devices:
- Smartphone (Android, iOS, any brand/model)
- Laptop (any brand/model)
- Tablet (any brand/model)
- Desktop computer
- Any device with internet access

Return your analysis as JSON:
{{
  "classification": "YES" | "NO" | "UNCLEAR",
  "confidence": 0.0-1.0,
  "extracted_info": {{"has_device": true/false, "device_type": "smartphone"|"laptop"|"tablet"|"other"|null}},
  "reasoning": "Brief explanation"
}}

Return ONLY the JSON object. No markdown, no explanation."""

def _get_commitment_prompt(question_text: str, user_input: str) -> str:
    """Generate prompt for commitment question classifier"""
    return f"""You are analyzing a user's response to an eligibility question about time commitment.

Context:
Question: {question_text}
User Response: "{user_input}"

Task: Classify whether the user can commit to ~2 hours per week for online teaching.

Classification Rules:
- YES: User agrees to 2+ hours per week (even if they commit more)
  Examples: "2 hrs should be fine", "Yes, I can do 3-4 hours", "More than 2 hrs", "Sure, that works", "Absolutely", "Yes", "I can do it"
- NO: User cannot commit 2 hours
  Examples: "Not possible", "Too busy", "Less than 2 hrs", "I can only do 1 hour", "Can't commit", "Too much time"
- UNCLEAR: Ambiguous or uncertain
  Examples: "Maybe", "Depends", "I'll try", "Not sure", "Probably"

Return your analysis as JSON:
{{
  "classification": "YES" | "NO" | "UNCLEAR",
  "confidence": 0.0-1.0,
  "extracted_info": {{"hours_per_week": number|null, "months": number|null}},
  "reasoning": "Brief explanation"
}}

Return ONLY the JSON object. No markdown, no explanation."""

def _get_consent_prompt(question_text: str, user_input: str) -> str:
    return f"""You are classifying consent/confirmation.

Context:
Question: {question_text}
User Response: "{user_input}"

Classify as YES if the user clearly agrees/consents, NO if clearly declines, otherwise UNCLEAR.

Return ONLY this JSON:
{{
  "classification": "YES" | "NO" | "UNCLEAR",
  "confidence": 0.0-1.0,
  "extracted_info": {{"agreed": true/false|null}},
  "reasoning": "Brief explanation"
}}"""

def _get_language_prompt(question_text: str, user_input: str) -> str:
    return f"""You are classifying language comfort from free text.

Context:
Question: {question_text}
User Response: "{user_input}"

Map comfort to one of: "comfortable", "ok", "uncomfortable".

Return ONLY this JSON:
{{
  "classification": "YES" | "NO" | "UNCLEAR",
  "confidence": 0.0-1.0,
  "extracted_info": {{"comfort_level": "comfortable"|"ok"|"uncomfortable"|null}},
  "reasoning": "Brief explanation"
}}"""

# ------------- RULE-BASED CLASSIFIERS (FAST PATH) -------------

def _rule_based_classify_age(user_input: str, context: Dict) -> Optional[ClassifyResponse]:
    """Rule-based age classifier - returns result if pattern matches, None otherwise"""
    txt = user_input.lower().strip()
    
    # Extract age if mentioned
    age_match = re.search(r'i am (\d+)|i\'m (\d+)|age (\d+)|(\d+) years old', txt)
    if age_match:
        age = int([g for g in age_match.groups() if g][0])
        if age >= 18:
            return ClassifyResponse(
                classification="YES",
                confidence=0.9,
                extracted_info={"is_18_plus": True, "age_number": age},
                reasoning=f"Rule-based: Extracted age: {age} years"
            )
        else:
            return ClassifyResponse(
                classification="NO",
                confidence=0.95,
                extracted_info={"is_18_plus": False, "age_number": age},
                reasoning=f"Rule-based: User is {age} years old (under 18)"
            )
    
    # Expanded yes patterns
    yes_patterns = ["yes", "yeah", "yep", "sure", "of course", "absolutely", "definitely", 
                    "ok", "okay", "sounds good", "works", "that's fine", "fine", "correct"]
    # Expanded no patterns
    no_patterns = ["no", "nope", "not really", "not yet", "not", "can't", "cant", "won't", "cannot"]
    
    # Check if it's a clear yes/no (must be standalone or at start/end)
    words = txt.split()
    if len(words) <= 3:  # Short responses like "yes", "yes I am", "I'm 19"
        if any(p in txt for p in yes_patterns) and not any(n in txt for n in no_patterns):
            return ClassifyResponse(
                classification="YES",
                confidence=0.85,
                extracted_info={"is_18_plus": True},
                reasoning="Rule-based: Positive response detected"
            )
        elif any(p in txt for p in no_patterns):
            return ClassifyResponse(
                classification="NO",
                confidence=0.85,
                extracted_info={"is_18_plus": False},
                reasoning="Rule-based: Negative response detected"
            )
    
    return None  # No clear pattern, needs LLM

def _rule_based_classify_device(user_input: str, context: Dict) -> Optional[ClassifyResponse]:
    """Rule-based device classifier - returns result if pattern matches, None otherwise"""
    txt = user_input.lower().strip()
    
    # Check for device mentions
    devices = ["smartphone", "phone", "laptop", "tablet", "desktop", "computer", "pc", "iphone", "android"]
    has_device = any(device in txt for device in devices)
    
    if has_device:
        device_type = "smartphone" if any(w in txt for w in ["phone", "smartphone", "iphone", "android"]) else \
                     ("laptop" if "laptop" in txt else \
                     ("tablet" if "tablet" in txt else "other"))
        return ClassifyResponse(
            classification="YES",
            confidence=0.85,
            extracted_info={"has_device": True, "device_type": device_type},
            reasoning="Rule-based: Device mentioned"
        )
    
    # Clear negative responses
    yes_patterns = ["yes", "yeah", "yep", "sure", "ok", "okay", "have one", "got it", "have it"]
    no_patterns = ["no", "nope", "don't", "dont", "don't have", "dont have", "not have", "no device"]
    
    words = txt.split()
    if len(words) <= 4:  # Short responses
        if any(p in txt for p in no_patterns):
            return ClassifyResponse(
                classification="NO",
                confidence=0.8,
                extracted_info={"has_device": False, "device_type": None},
                reasoning="Rule-based: No device indicated"
            )
        elif any(p in txt for p in yes_patterns):
            return ClassifyResponse(
                classification="YES",
                confidence=0.75,
                extracted_info={"has_device": True, "device_type": None},
                reasoning="Rule-based: Affirmative response"
            )
    
    return None  # No clear pattern, needs LLM

def _rule_based_classify_commitment(user_input: str, context: Dict) -> Optional[ClassifyResponse]:
    """Rule-based commitment classifier - returns result if pattern matches, None otherwise"""
    txt = user_input.lower().strip()
    
    # Extract hours and months if mentioned
    hour_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', txt)
    months_match = re.search(r'(\d+(?:\.\d+)?)\s*months?', txt)
    if hour_match:
        hours = float(hour_match.group(1))
        months_val = float(months_match.group(1)) if months_match else None
        if hours >= 2:
            return ClassifyResponse(
                classification="YES",
                confidence=0.9,
                extracted_info={"hours_per_week": hours, "months": months_val},
                reasoning=f"Rule-based: {hours} hours/week detected (meets requirement)"
            )
        else:
            return ClassifyResponse(
                classification="NO",
                confidence=0.85,
                extracted_info={"hours_per_week": hours, "months": months_val},
                reasoning=f"Rule-based: Only {hours} hours/week (less than required 2 hours)"
            )
    
    # Expanded yes patterns
    yes_patterns = ["yes", "yeah", "yep", "sure", "of course", "absolutely", "can do", "2 hrs", 
                    "2 hours", "fine", "ok", "okay", "works", "that works", "sounds good"]
    # Expanded no patterns
    no_patterns = ["no", "not possible", "too busy", "can't", "cant", "won't", "cannot", 
                   "less than", "less than 2", "not enough time", "too much"]
    
    words = txt.split()
    if len(words) <= 5:  # Short responses
        if any(p in txt for p in yes_patterns) and not any(n in txt for n in no_patterns):
            return ClassifyResponse(
                classification="YES",
                confidence=0.75,
                extracted_info={"hours_per_week": None, "months": None},
                reasoning="Rule-based: Positive commitment response"
            )
        elif any(p in txt for p in no_patterns):
            return ClassifyResponse(
                classification="NO",
                confidence=0.8,
                extracted_info={"hours_per_week": None, "months": None},
                reasoning="Rule-based: Negative commitment response"
            )
    
    return None  # No clear pattern, needs LLM

def _rule_based_classify_consent(user_input: str, context: Dict) -> Optional[ClassifyResponse]:
    """Rule-based consent classifier - returns result if pattern matches, None otherwise"""
    txt = user_input.lower().strip()
    
    yes_patterns = ["yes", "yeah", "yep", "ok", "okay", "sure", "agree", "agreed", "confirmed", 
                    "correct", "right", "sounds good", "works", "fine", "alright"]
    no_patterns = ["no", "nope", "don't", "dont", "decline", "later", "not now", "not ready"]
    
    words = txt.split()
    if len(words) <= 3:  # Short responses
        if any(p in txt for p in yes_patterns) and not any(n in txt for n in no_patterns):
            return ClassifyResponse(
                classification="YES",
                confidence=0.85,
                extracted_info={"agreed": True},
                reasoning="Rule-based: Positive consent"
            )
        elif any(p in txt for p in no_patterns):
            return ClassifyResponse(
                classification="NO",
                confidence=0.85,
                extracted_info={"agreed": False},
                reasoning="Rule-based: Negative consent"
            )
    
    return None  # No clear pattern, needs LLM

def _rule_based_classify_language(user_input: str, context: Dict) -> Optional[ClassifyResponse]:
    """Rule-based language comfort classifier - returns result if pattern matches, None otherwise"""
    txt = user_input.lower().strip()
    
    comfortable_patterns = ["very", "very comfortable", "comfortable", "good", "fine", "ok", "okay", "yes"]
    uncomfortable_patterns = ["not", "hard", "difficult", "uncomfortable", "struggle", "can't", "cant"]
    
    words = txt.split()
    if len(words) <= 4:  # Short responses
        if any(p in txt for p in comfortable_patterns) and not any(u in txt for u in uncomfortable_patterns):
            comfort_level = "comfortable" if "very" in txt or "comfortable" in txt else "ok"
            return ClassifyResponse(
                classification="YES",
                confidence=0.75,
                extracted_info={"comfort_level": comfort_level},
                reasoning="Rule-based: Comfortable with language"
            )
        elif any(p in txt for p in uncomfortable_patterns):
            return ClassifyResponse(
                classification="NO",
                confidence=0.75,
                extracted_info={"comfort_level": "uncomfortable"},
                reasoning="Rule-based: Uncomfortable with language"
            )
    
    return None  # No clear pattern, needs LLM

# ------------- FALLBACK CLASSIFIERS -------------

def _fallback_classify_age(user_input: str, context: Dict, error: str) -> ClassifyResponse:
    """Fallback rule-based age classifier when LLM fails"""
    txt = user_input.lower().strip()
    
    # Extract age if mentioned
    age_match = re.search(r'i am (\d+)|i\'m (\d+)|age (\d+)|(\d+) years old', txt)
    if age_match:
        age = int([g for g in age_match.groups() if g][0])
        if age >= 18:
            return ClassifyResponse(
                classification="YES",
                confidence=0.9,
                extracted_info={"is_18_plus": True, "age_number": age},
                reasoning=f"Extracted age: {age} years"
            )
        else:
            return ClassifyResponse(
                classification="NO",
                confidence=0.95,
                extracted_info={"is_18_plus": False, "age_number": age},
                reasoning=f"User is {age} years old (under 18)"
            )
    
    # Simple pattern matching
    yes_patterns = ["yes", "yeah", "sure", "of course", "absolutely", "yep"]
    no_patterns = ["no", "not yet", "nope", "not really"]
    
    if any(p in txt for p in yes_patterns):
        return ClassifyResponse(
            classification="YES",
            confidence=0.7,
            extracted_info={"is_18_plus": True},
            reasoning=f"Fallback: Positive response detected"
        )
    elif any(p in txt for p in no_patterns):
        return ClassifyResponse(
            classification="NO",
            confidence=0.7,
            extracted_info={"is_18_plus": False},
            reasoning=f"Fallback: Negative response detected"
        )
    
    return ClassifyResponse(
        classification="UNCLEAR",
        confidence=0.5,
        extracted_info={},
        reasoning=f"LLM failed: {error}. Could not classify input."
    )

def _fallback_classify_device(user_input: str, context: Dict, error: str) -> ClassifyResponse:
    """Fallback rule-based device classifier when LLM fails"""
    txt = user_input.lower().strip()
    
    # Check for device mentions
    devices = ["smartphone", "phone", "laptop", "tablet", "desktop", "computer", "pc", "iphone", "android"]
    has_device = any(device in txt for device in devices)
    
    if has_device:
        return ClassifyResponse(
            classification="YES",
            confidence=0.75,
            extracted_info={"has_device": True, "device_type": ("smartphone" if "phone" in txt or "smartphone" in txt or "iphone" in txt or "android" in txt else ("laptop" if "laptop" in txt else ("tablet" if "tablet" in txt else "other")))},
            reasoning=f"Fallback: Device mentioned"
        )
    elif any(word in txt for word in ["no", "don't", "dont", "not"]):
        return ClassifyResponse(
            classification="NO",
            confidence=0.7,
            extracted_info={"has_device": False, "device_type": None},
            reasoning=f"Fallback: No device mentioned"
        )
    
    return ClassifyResponse(
        classification="UNCLEAR",
        confidence=0.5,
        extracted_info={},
        reasoning=f"LLM failed: {error}. Could not classify input."
    )

def _fallback_classify_commitment(user_input: str, context: Dict, error: str) -> ClassifyResponse:
    """Fallback rule-based commitment classifier when LLM fails"""
    txt = user_input.lower().strip()
    
    # Extract hours and months if mentioned
    hour_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', txt)
    months_match = re.search(r'(\d+(?:\.\d+)?)\s*months?', txt)
    if hour_match:
        hours = float(hour_match.group(1))
        if hours >= 2:
            return ClassifyResponse(
                classification="YES",
                confidence=0.85,
                extracted_info={"hours_per_week": hours, "months": float(months_match.group(1)) if months_match else None},
                reasoning=f"Fallback: {hours} hours/week detected"
            )
        else:
            return ClassifyResponse(
                classification="NO",
                confidence=0.8,
                extracted_info={"hours_per_week": hours, "months": float(months_match.group(1)) if months_match else None},
                reasoning=f"Fallback: Only {hours} hours/week (less than required 2 hours)"
            )
    
    # Simple pattern matching
    yes_patterns = ["yes", "fine", "okay", "sure", "of course", "absolutely", "can do", "2 hrs"]
    no_patterns = ["no", "not possible", "too busy", "can't", "cant"]
    
    if any(p in txt for p in yes_patterns):
        return ClassifyResponse(
            classification="YES",
            confidence=0.65,
            extracted_info={"hours_per_week": None, "months": None},
            reasoning=f"Fallback: Positive response detected"
        )
    elif any(p in txt for p in no_patterns):
        return ClassifyResponse(
            classification="NO",
            confidence=0.7,
            extracted_info={"hours_per_week": None, "months": None},
            reasoning=f"Fallback: Negative response detected"
        )
    
    return ClassifyResponse(
        classification="UNCLEAR",
        confidence=0.5,
        extracted_info={},
        reasoning=f"LLM failed: {error}. Could not classify input."
    )

# ------------- CLASSIFIER FUNCTIONS -------------

async def _classify_age(user_input: str, context: Dict) -> ClassifyResponse:
    """Classify age-related response - Hybrid: rule-based first, then LLM"""
    # Step 1: Try rule-based first (fast, cost-free)
    rule_result = _rule_based_classify_age(user_input, context)
    if rule_result:
        print(f"[llm.classify_response] Age: Rule-based match, skipping LLM")
        return rule_result
    
    # Step 2: Rule-based didn't match, use LLM for complex responses
    print(f"[llm.classify_response] Age: No rule-based match, calling LLM")
    question_text = context.get("question_text", "Are you 18 or older?")
    
    prompt = _get_age_prompt(question_text, user_input)
    
    data, error = await call_llm_for_json(
        prompt=prompt,
        temperature=0.2,  # Low temperature for more consistent classification
        max_tokens=200
    )
    
    if error:
        # Fallback to rule-based classification
        return _fallback_classify_age(user_input, context, error)
    
    return ClassifyResponse(
        classification=data.get("classification", "UNCLEAR"),
        confidence=data.get("confidence", 0.5),
        extracted_info=data.get("extracted_info", {}),
        reasoning=data.get("reasoning", "No reasoning provided")
    )

async def _classify_device(user_input: str, context: Dict) -> ClassifyResponse:
    """Classify device-related response - Hybrid: rule-based first, then LLM"""
    # Step 1: Try rule-based first (fast, cost-free)
    rule_result = _rule_based_classify_device(user_input, context)
    if rule_result:
        print(f"[llm.classify_response] Device: Rule-based match, skipping LLM")
        return rule_result
    
    # Step 2: Rule-based didn't match, use LLM for complex responses
    print(f"[llm.classify_response] Device: No rule-based match, calling LLM")
    question_text = context.get("question_text", "Do you have a smartphone or laptop?")
    
    prompt = _get_device_prompt(question_text, user_input)
    
    data, error = await call_llm_for_json(
        prompt=prompt,
        temperature=0.2,
        max_tokens=200
    )
    
    if error:
        # Fallback to rule-based classification
        return _fallback_classify_device(user_input, context, error)
    
    return ClassifyResponse(
        classification=data.get("classification", "UNCLEAR"),
        confidence=data.get("confidence", 0.5),
        extracted_info=data.get("extracted_info", {}),
        reasoning=data.get("reasoning", "No reasoning provided")
    )

async def _classify_commitment(user_input: str, context: Dict) -> ClassifyResponse:
    """Classify commitment-related response - Hybrid: rule-based first, then LLM"""
    # Step 1: Try rule-based first (fast, cost-free)
    rule_result = _rule_based_classify_commitment(user_input, context)
    if rule_result:
        print(f"[llm.classify_response] Commitment: Rule-based match, skipping LLM")
        return rule_result
    
    # Step 2: Rule-based didn't match, use LLM for complex responses
    print(f"[llm.classify_response] Commitment: No rule-based match, calling LLM")
    question_text = context.get("question_text", "Can you commit 2 hours per week?")
    
    prompt = _get_commitment_prompt(question_text, user_input)
    
    data, error = await call_llm_for_json(
        prompt=prompt,
        temperature=0.2,
        max_tokens=200
    )
    
    if error:
        # Fallback to rule-based classification
        return _fallback_classify_commitment(user_input, context, error)
    
    return ClassifyResponse(
        classification=data.get("classification", "UNCLEAR"),
        confidence=data.get("confidence", 0.5),
        extracted_info=data.get("extracted_info", {}),
        reasoning=data.get("reasoning", "No reasoning provided")
    )

async def _classify_consent(user_input: str, context: Dict) -> ClassifyResponse:
    """Classify consent-related response - Hybrid: rule-based first, then LLM"""
    # Step 1: Try rule-based first (fast, cost-free)
    rule_result = _rule_based_classify_consent(user_input, context)
    if rule_result:
        print(f"[llm.classify_response] Consent: Rule-based match, skipping LLM")
        return rule_result
    
    # Step 2: Rule-based didn't match, use LLM for complex responses
    print(f"[llm.classify_response] Consent: No rule-based match, calling LLM")
    question_text = context.get("question_text", "Do you agree?")
    prompt = _get_consent_prompt(question_text, user_input)
    data, error = await call_llm_for_json(prompt=prompt, temperature=0.2, max_tokens=120)
    if error:
        # Fallback to rule-based classification
        return _fallback_classify_consent(user_input, context, error)
    return ClassifyResponse(
        classification=data.get("classification", "UNCLEAR"),
        confidence=data.get("confidence", 0.5),
        extracted_info=data.get("extracted_info", {}),
        reasoning=data.get("reasoning", "No reasoning provided")
    )

async def _classify_language(user_input: str, context: Dict) -> ClassifyResponse:
    """Classify language comfort-related response - Hybrid: rule-based first, then LLM"""
    # Step 1: Try rule-based first (fast, cost-free)
    rule_result = _rule_based_classify_language(user_input, context)
    if rule_result:
        print(f"[llm.classify_response] Language: Rule-based match, skipping LLM")
        return rule_result
    
    # Step 2: Rule-based didn't match, use LLM for complex responses
    print(f"[llm.classify_response] Language: No rule-based match, calling LLM")
    question_text = context.get("question_text", "Are you comfortable with this language?")
    prompt = _get_language_prompt(question_text, user_input)
    data, error = await call_llm_for_json(prompt=prompt, temperature=0.2, max_tokens=120)
    if error:
        # Fallback to rule-based classification
        txt = user_input.lower().strip()
        if any(w in txt for w in ["very", "comfortable", "good", "fine", "ok", "okay"]):
            return ClassifyResponse(classification="YES", confidence=0.6, extracted_info={"comfort_level": "comfortable" if "very" in txt or "comfortable" in txt else "ok"}, reasoning="Fallback heuristic")
        if any(w in txt for w in ["not", "hard", "difficult", "uncomfortable"]):
            return ClassifyResponse(classification="NO", confidence=0.6, extracted_info={"comfort_level": "uncomfortable"}, reasoning="Fallback heuristic")
        return ClassifyResponse(classification="UNCLEAR", confidence=0.5, extracted_info={"comfort_level": None}, reasoning=f"LLM failed: {error}")
    return ClassifyResponse(
        classification=data.get("classification", "UNCLEAR"),
        confidence=data.get("confidence", 0.5),
        extracted_info=data.get("extracted_info", {}),
        reasoning=data.get("reasoning", "No reasoning provided")
    )

# ------------- MAIN ENDPOINT -------------

@router.post("/llm.classify_response", response_model=ClassifyResponse)
async def classify_response(body: ClassifyRequest):
    """
    Classify a user's response to an eligibility question using LLM.
    
    This tool uses Claude to intelligently understand natural language responses
    to eligibility questions, reducing false negatives and improving UX.
    
    Supports question types:
    - consent: User agrees/declines
    - age: Confirm user is 18+
    - device: Confirm user has suitable device
    - commitment: Confirm user can commit 2+ hours/week
    - language_comfort: Comfort level mapping
    """
    print(f"[llm.classify_response] Question type: {body.question_type}, User input: '{body.user_input}'")
    
    try:
        # Route to appropriate classifier
        if body.question_type == "consent":
            result = await _classify_consent(body.user_input, body.context)
        elif body.question_type == "age":
            result = await _classify_age(body.user_input, body.context)
        elif body.question_type == "device":
            result = await _classify_device(body.user_input, body.context)
        elif body.question_type == "commitment":
            result = await _classify_commitment(body.user_input, body.context)
        elif body.question_type == "language_comfort":
            result = await _classify_language(body.user_input, body.context)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown question type: {body.question_type}"
            )
        
        print(f"[llm.classify_response] Result: {result.classification} (confidence={result.confidence:.2f})")
        return result
        
    except Exception as e:
        print(f"[llm.classify_response] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

