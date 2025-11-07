"""
LLM-powered smart editing for teaching preferences.

This tool enables context-aware editing of teaching preferences, understanding
natural language edit requests with conversation context.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Dict, List, Any, Optional, Union
import json

from tools.llm_core import call_llm_for_json

router = APIRouter()

# ------------- REQUEST/RESPONSE MODELS -------------

class SmartEditRequest(BaseModel):
    """Request to intelligently edit teaching preferences"""
    conversation_history: List[str] = Field(
        ...,
        description="List of conversation messages (user/assistant alternating)"
    )
    current_profile: Dict[str, Any] = Field(
        ...,
        description="Current teaching profile to edit"
    )
    user_input: str = Field(
        ...,
        description="User's edit request text"
    )

class SmartEditResponse(BaseModel):
    """Response with updated profile"""
    understood: bool = Field(
        ...,
        description="Whether the edit request was understood"
    )
    updated_subjects: List[str] = Field(
        ...,
        description="Updated subjects list"
    )
    updated_grades: str = Field(
        ...,
        description="Updated grades (e.g., '6-8', '9-10', '6')"
    )
    updated_language: str = Field(
        ...,
        description="Updated language/medium"
    )
    explanation: str = Field(
        ...,
        description="Explanation of what changed"
    )
    
    @field_validator('updated_grades', mode='before')
    @classmethod
    def coerce_grades_to_string(cls, v):
        """Convert any grade value (int or str) to string"""
        if isinstance(v, (int, float)):
            return str(v)
        return str(v) if v else ""

# ------------- PROMPT TEMPLATE -------------

def _get_smart_edit_prompt(
    conversation_history: List[str],
    current_profile: Dict[str, Any],
    user_input: str
) -> str:
    """Generate prompt for smart editing"""
    
    # Format conversation history
    history_formatted = "\n".join([f"- {msg}" for msg in conversation_history[-6:]])  # Last 6 messages
    
    # Extract current profile fields
    subjects = current_profile.get("subjects", [])
    grades = current_profile.get("grades", "")
    language = current_profile.get("language", "")
    
    prompt = f"""You are helping a volunteer update their teaching preferences.

CONVERSATION HISTORY (last few messages):
{history_formatted}

CURRENT PROFILE:
- Subjects: {subjects}
- Grades: {grades}
- Language: {language}

USER'S EDIT REQUEST: "{user_input}"

YOUR TASK:
1. Understand what the user wants to change based on the conversation context
2. Update only the fields that need changing
3. Keep other fields as they are (unless explicitly changing them)

EXAMPLES OF EDITS:
- "Change to English" → Update language field to "English"
- "Add Science" → Add "Science" to subjects list (keep existing subjects)
- "Make it Grade 7-8" or "Change grades to 9-10" → Update grades field
- "Update language to Kannada" → Update language field to "Kannada"
- "Remove Math" → Remove "Math" from subjects (keep others)
- "Change to Science only" → Replace subjects with just "Science"

VALID VALUES:
- Subjects: Math, English, Science, Social Studies, Computer Science, Hindi
- Grades: "1-3", "4-5", "6-8", "9-10", "11-12", or single grades as strings ("1", "2", "6", etc.)
- Language: Hindi, English, Tamil, Kannada, Telugu, Marathi, Bengali, Gujarati, Both

CRITICAL: Always return grades as a STRING (e.g., "6", "6-8", "9-10"), never as a number

IMPORTANT RULES:
- If user wants to ADD a subject, include it in the list (keep existing subjects)
- If user wants to REPLACE subjects, provide only the new subjects
- If user wants to REMOVE a subject, exclude it from the list
- If user wants to CHANGE language or grades, update that field
- Keep fields that were not mentioned in the edit unchanged
- If the request is ambiguous or unclear, set "understood": false

Return ONLY valid JSON (no markdown, no extra text):
{{
    "understood": true or false,
    "updated_subjects": ["updated list or keep existing"],
    "updated_grades": "updated grade or keep existing",
    "updated_language": "updated language or keep existing",
    "explanation": "brief explanation of what changed or why not understood"
}}"""
    
    return prompt

# ------------- RULE-BASED EDIT HANDLER (FAST PATH) -------------

def _rule_based_handle_edit(
    current_profile: Dict[str, Any],
    user_input: str
) -> Optional[SmartEditResponse]:
    """
    Rule-based edit handler - returns result if pattern matches, None otherwise.
    Handles simple, common edit patterns without LLM.
    """
    txt = user_input.lower().strip()
    current_subjects = current_profile.get("subjects", [])
    current_grades = current_profile.get("grades", "")
    current_language = current_profile.get("language", "")
    
    # Language change patterns
    language_map = {
        "english": "English", "hindi": "Hindi", "tamil": "Tamil", "kannada": "Kannada",
        "telugu": "Telugu", "marathi": "Marathi", "bengali": "Bengali", "gujarati": "Gujarati"
    }
    
    # Pattern: "change to [language]", "update language to [language]", "make it [language]"
    for lang_key, lang_value in language_map.items():
        patterns = [
            f"change to {lang_key}",
            f"update to {lang_key}",
            f"make it {lang_key}",
            f"language to {lang_key}",
            f"update language to {lang_key}",
            f"change language to {lang_key}"
        ]
        if any(p in txt for p in patterns) or (("change" in txt or "update" in txt) and lang_key in txt and "language" not in txt):
            return SmartEditResponse(
                understood=True,
                updated_subjects=current_subjects,
                updated_grades=current_grades,
                updated_language=lang_value,
                explanation=f"Changed language to {lang_value}"
            )
    
    # Pattern: "add [subject]"
    valid_subjects = ["Math", "English", "Science", "Social Studies", "Computer Science", "Hindi"]
    for subject in valid_subjects:
        if f"add {subject.lower()}" in txt or (txt.startswith("add ") and subject.lower() in txt):
            new_subjects = current_subjects.copy()
            if subject not in new_subjects:
                new_subjects.append(subject)
            return SmartEditResponse(
                understood=True,
                updated_subjects=new_subjects,
                updated_grades=current_grades,
                updated_language=current_language,
                explanation=f"Added {subject} to subjects"
            )
    
    # Pattern: "remove [subject]"
    for subject in valid_subjects:
        if f"remove {subject.lower()}" in txt or (txt.startswith("remove ") and subject.lower() in txt):
            new_subjects = [s for s in current_subjects if s != subject]
            return SmartEditResponse(
                understood=True,
                updated_subjects=new_subjects,
                updated_grades=current_grades,
                updated_language=current_language,
                explanation=f"Removed {subject} from subjects"
            )
    
    # Pattern: grade changes - "grade [X]", "grade [X-Y]", "make it grade [X]"
    grade_patterns = {
        "1-3": ["1-3", "1 to 3", "grade 1-3", "grade 1 to 3"],
        "4-5": ["4-5", "4 to 5", "grade 4-5", "grade 4 to 5"],
        "6-8": ["6-8", "6 to 8", "grade 6-8", "grade 6 to 8", "grade 6", "grade 7", "grade 8"],
        "9-10": ["9-10", "9 to 10", "grade 9-10", "grade 9 to 10", "grade 9", "grade 10"],
        "11-12": ["11-12", "11 to 12", "grade 11-12", "grade 11 to 12", "grade 11", "grade 12"]
    }
    
    for grade, patterns_list in grade_patterns.items():
        for pattern in patterns_list:
            if pattern in txt or (("grade" in txt or "make it" in txt) and any(p in txt for p in patterns_list)):
                return SmartEditResponse(
                    understood=True,
                    updated_subjects=current_subjects,
                    updated_grades=grade,
                    updated_language=current_language,
                    explanation=f"Changed grades to {grade}"
                )
    
    # Pattern: "change to [subject]" or "make it [subject]" (replace subjects)
    for subject in valid_subjects:
        if (f"change to {subject.lower()}" in txt and "language" not in txt) or \
           (txt.startswith("change to ") and subject.lower() in txt and "language" not in txt) or \
           (f"make it {subject.lower()}" in txt and "language" not in txt):
            return SmartEditResponse(
                understood=True,
                updated_subjects=[subject],
                updated_grades=current_grades,
                updated_language=current_language,
                explanation=f"Changed subjects to {subject}"
            )
    
    return None  # No clear pattern, needs LLM

# ------------- LLM HANDLER -------------

async def _handle_smart_edit(
    conversation_history: List[str],
    current_profile: Dict[str, Any],
    user_input: str
) -> SmartEditResponse:
    """Handle smart edit request using LLM"""
    
    prompt = _get_smart_edit_prompt(conversation_history, current_profile, user_input)
    
    data, error = await call_llm_for_json(
        prompt=prompt,
        temperature=0.2,  # Low temperature for more consistent results
        max_tokens=300
    )
    
    if error:
        # Fallback: return unchanged profile
        return SmartEditResponse(
            understood=False,
            updated_subjects=current_profile.get("subjects", []),
            updated_grades=current_profile.get("grades", ""),
            updated_language=current_profile.get("language", ""),
            explanation=f"LLM call failed: {error}"
        )
    
    # Extract response fields
    understood = data.get("understood", False)
    updated_subjects = data.get("updated_subjects", current_profile.get("subjects", []))
    updated_grades = data.get("updated_grades", current_profile.get("grades", ""))
    updated_language = data.get("updated_language", current_profile.get("language", ""))
    explanation = data.get("explanation", "No changes made")
    
    return SmartEditResponse(
        understood=understood,
        updated_subjects=updated_subjects,
        updated_grades=updated_grades,
        updated_language=updated_language,
        explanation=explanation
    )

# ------------- MAIN ENDPOINT -------------

@router.post("/llm.handle_smart_edit", response_model=SmartEditResponse)
async def handle_smart_edit(body: SmartEditRequest):
    """
    Intelligently handle teaching preference editing based on natural language.
    
    This tool uses LLM to understand context-aware edit requests like:
    - "Change to English" → Updates language field
    - "Add Science" → Adds Science to subjects
    - "Make it Grade 7-8" → Updates grades
    - "Update language to Kannada" → Updates language field
    """
    print(f"[llm.handle_smart_edit] User input: '{body.user_input}'")
    print(f"[llm.handle_smart_edit] Current profile: {body.current_profile}")
    
    try:
        # Step 1: Try rule-based first (fast, cost-free)
        rule_result = _rule_based_handle_edit(body.current_profile, body.user_input)
        if rule_result:
            print(f"[llm.handle_smart_edit] Rule-based match, skipping LLM")
            return rule_result
        
        # Step 2: Rule-based didn't match, use LLM for complex edits
        print(f"[llm.handle_smart_edit] No rule-based match, calling LLM")
        result = await _handle_smart_edit(
            body.conversation_history,
            body.current_profile,
            body.user_input
        )
        
        print(f"[llm.handle_smart_edit] Understood: {result.understood}, Explanation: {result.explanation}")
        return result
        
    except Exception as e:
        print(f"[llm.handle_smart_edit] Error: {e}")
        
        # Return unchanged profile on error
        return SmartEditResponse(
            understood=False,
            updated_subjects=body.current_profile.get("subjects", []),
            updated_grades=body.current_profile.get("grades", ""),
            updated_language=body.current_profile.get("language", ""),
            explanation=f"Error processing edit request: {str(e)}"
        )

