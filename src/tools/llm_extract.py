"""
Profile Extraction Tool - Hybrid approach (rule-based + LLM)
"""
from pydantic import BaseModel, Field
from fastapi import APIRouter
from typing import List, Dict, Optional
from .llm_core import call_llm_for_json

router = APIRouter()

# ------------- MODEL DEFINITIONS -------------

class ExtractInput(BaseModel):
    state: str
    text: str
    locale: str = "en-IN"
    catalog: Dict[str, List[str]] = Field(default_factory=dict)
    use_llm: bool = True  # Toggle LLM enhancement

class ExtractOutput(BaseModel):
    subjects: List[str] = []
    grades: List[str] = []
    availability_note: Optional[str] = None
    timezone: Optional[str] = None
    confidence: float = 0.0
    notes: List[str] = []
    extraction_method: str = "rule_based"  # "rule_based", "llm", or "hybrid"
    extracted_something: bool = False  # Helper for agent to know if extraction succeeded

# ------------- RULE-BASED EXTRACTION (FAST) -------------

def _fast_extract(text: str, catalog: Dict[str, List[str]]) -> tuple[List[str], List[str], Optional[str], float]:
    """
    Fast rule-based extraction using string matching.
    Returns: (subjects, grades, availability, confidence)
    """
    txt = text.lower()
    
    # Extract subjects
    subjects = []
    subject_variants = {
        "math": ["math", "maths", "mathematics"],
        "english": ["english", "eng"],
        "science": ["science", "sci"],
        "social studies": ["social", "history", "geography", "civics"],
        "computer science": ["computer", "coding", "programming", "cs"],
        "hindi": ["hindi"],
        "physics": ["physics"],
        "chemistry": ["chemistry", "chem"],
        "biology": ["biology", "bio"]
    }
    
    for subject in catalog.get("subjects", []):
        subject_lower = subject.lower()
        variants = subject_variants.get(subject_lower, [subject_lower])
        if any(variant in txt for variant in variants):
            subjects.append(subject)
    
    # Extract grades
    grades = []
    grade_patterns = {
        "1-3": ["1-3", "1 to 3", "class 1", "class 2", "class 3", "primary", "lower primary"],
        "4-5": ["4-5", "4 to 5", "class 4", "class 5", "upper primary"],
        "6-8": ["6-8", "6 to 8", "class 6", "class 7", "class 8", "middle school", "6th", "7th", "8th"],
        "9-10": ["9-10", "9 to 10", "class 9", "class 10", "9th", "10th", "high school"],
        "11-12": ["11-12", "11 to 12", "class 11", "class 12", "11th", "12th", "senior"]
    }
    
    for grade in catalog.get("grades", []):
        patterns = grade_patterns.get(grade, [grade])
        if any(pattern in txt for pattern in patterns):
            grades.append(grade)
    
    # Extract availability
    availability = None
    if any(word in txt for word in ["weekend", "saturday", "sunday"]):
        availability = "weekends"
    elif any(word in txt for word in ["evening", "night", "after"]):
        availability = "weekday evenings"
    elif any(word in txt for word in ["morning"]):
        availability = "mornings"
    elif any(word in txt for word in ["afternoon"]):
        availability = "afternoons"
    
    # Calculate confidence based on what was extracted
    confidence = 0.0
    if subjects:
        confidence += 0.4
    if grades:
        confidence += 0.3
    if availability:
        confidence += 0.3
    
    return subjects, grades, availability, confidence

# ------------- LLM ENHANCEMENT -------------

async def _llm_enhance(
    text: str,
    catalog: Dict[str, List[str]],
    fast_results: tuple[List[str], List[str], Optional[str], float]
) -> ExtractOutput:
    """
    Use LLM to enhance or validate fast extraction results.
    """
    subjects, grades, availability, _ = fast_results
    
    prompt = f"""Extract profile information from this text. Use the provided catalog to match subjects and grades.

Text: "{text}"

Available subjects: {catalog.get("subjects", [])}
Available grades: {catalog.get("grades", [])}

Fast extraction found:
- Subjects: {subjects if subjects else "none"}
- Grades: {grades if grades else "none"}
- Availability: {availability if availability else "none"}

Instructions:
1. Validate the fast extraction results
2. Find any missed subjects/grades (handle variations: "maths"→"Math", "middle school"→"6-8")
3. Extract more detailed availability information
4. Identify timezone if mentioned
5. Add any relevant notes

Return JSON:
{{
  "subjects": ["list of subjects - validate and add missing ones"],
  "grades": ["list of grades - validate and add missing ones"],
  "availability_note": "detailed availability pattern",
  "timezone": "timezone if mentioned (e.g., 'Asia/Kolkata') or null",
  "confidence": 0.0-1.0,
  "notes": ["any additional insights or clarifications"]
}}
"""

    data, error = await call_llm_for_json(
        prompt=prompt,
        temperature=0,
        max_tokens=300
    )
    
    if error:
        # LLM failed, return fast results
        return ExtractOutput(
            subjects=subjects,
            grades=grades,
            availability_note=availability,
            confidence=fast_results[3],
            extraction_method="rule_based_only"
        )
    
    # Merge LLM results with fast results (LLM takes precedence)
    return ExtractOutput(
        subjects=data.get("subjects", subjects),
        grades=data.get("grades", grades),
        availability_note=data.get("availability_note", availability),
        timezone=data.get("timezone"),
        confidence=data.get("confidence", 0.8),
        notes=data.get("notes", []),
        extraction_method="hybrid"
    )

# ------------- MCP ENDPOINT -------------

@router.post("/llm.extract_profile_fields", response_model=ExtractOutput)
async def extract_profile_fields(body: ExtractInput):
    """
    MCP Tool: llm.extract_profile_fields (Hybrid)
    
    Extracts profile information using a two-stage approach:
    1. Fast rule-based extraction (< 1ms) - handles obvious cases
    2. LLM enhancement (optional) - validates and fills gaps
    
    Examples:
    
    Fast path (rule-based only):
    Input: "I teach math and english"
    Output: {"subjects": ["Math", "English"], "method": "rule_based"}
    
    Hybrid (rule-based + LLM):
    Input: "I'm good at teaching middle school students mathematics and natural sciences on weekends"
    Fast: {"subjects": ["Math", "Science"], "grades": ["6-8"]}
    LLM enhances: Validates + adds availability details
    Output: {"subjects": ["Math", "Science"], "grades": ["6-8"], "availability": "weekends", "method": "hybrid"}
    
    Set use_llm=false to skip LLM enhancement (faster, less accurate).
    """
    print(f"[llm.extract_profile_fields] Text: '{body.text}', State: {body.state}, Use LLM: {body.use_llm}")
    
    # Stage 1: Fast rule-based extraction
    fast_results = _fast_extract(body.text, body.catalog)
    subjects, grades, availability, confidence = fast_results
    
    print(f"[llm.extract_profile_fields] Fast extraction: subjects={subjects}, grades={grades}, confidence={confidence}")
    
    # If LLM disabled or decent confidence, return fast results
    # Threshold: 0.4 is good enough if we found something concrete
    if not body.use_llm or confidence >= 0.4 or (subjects or grades):
        return ExtractOutput(
            subjects=subjects,
            grades=grades,
            availability_note=availability,
            confidence=max(confidence, 0.7) if (subjects or grades) else confidence,
            extraction_method="rule_based",
            extracted_something=bool(subjects or grades or availability)
        )
    
    # Stage 2: LLM enhancement only for truly ambiguous cases (nothing extracted)
    print(f"[llm.extract_profile_fields] No clear extraction (confidence={confidence}), calling LLM for enhancement")
    enhanced = await _llm_enhance(body.text, body.catalog, fast_results)
    
    print(f"[llm.extract_profile_fields] LLM enhanced: subjects={enhanced.subjects}, confidence={enhanced.confidence}")
    
    enhanced.extracted_something = bool(enhanced.subjects or enhanced.grades or enhanced.availability_note)
    return enhanced
