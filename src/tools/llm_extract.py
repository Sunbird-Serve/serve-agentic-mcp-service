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
    language_medium: Optional[str] = None  # Hindi, English, Tamil, etc.
    timezone: Optional[str] = None
    confidence: float = 0.0
    notes: List[str] = []
    extraction_method: str = "rule_based"  # "rule_based", "llm", or "hybrid"
    extracted_something: bool = False  # Helper for agent to know if extraction succeeded

# ------------- RULE-BASED EXTRACTION (FAST) -------------

def _fast_extract(text: str, catalog: Dict[str, List[str]]) -> tuple[List[str], List[str], Optional[str], Optional[str], float]:
    """
    Fast rule-based extraction using string matching.
    Returns: (subjects, grades, availability, language_medium, confidence)
    """
    txt = text.lower()
    
    # Extract subjects with enhanced variants
    subjects = []
    subject_variants = {
        "math": ["math", "maths", "mathematics", "arithmetic", "numbers", "calculation", "algebra", "geometry"],
        "english": ["english", "eng", "language", "grammar", "literature", "writing", "reading"],
        "science": ["science", "sci", "natural science", "physics", "chemistry", "biology"],
        "social studies": ["social", "history", "geography", "civics", "political"],
        "computer science": ["computer", "coding", "programming", "cs", "technology", "software"],
        "hindi": ["hindi", "हिंदी"],
        "physics": ["physics"],
        "chemistry": ["chemistry", "chem"],
        "biology": ["biology", "bio", "life science"]
    }
    
    for subject in catalog.get("subjects", []):
        subject_lower = subject.lower()
        variants = subject_variants.get(subject_lower, [subject_lower])
        if any(variant in txt for variant in variants):
            if subject not in subjects:  # Avoid duplicates
                subjects.append(subject)
    
    # Extract grades with enhanced patterns
    grades = []
    
    # Special case: "any grade", "all grades", or vague like "higher grade"
    if any(phrase in txt for phrase in ["any grade", "all grade", "flexible", "open to all"]):
        grades = catalog.get("grades", [])
    elif "higher grade" in txt or "senior grade" in txt or "advanced" in txt:
        # Higher grades typically mean 9-10 or 11-12
        grades = [g for g in catalog.get("grades", []) if g in ["9-10", "11-12"]]
    else:
        grade_patterns = {
            "1-3": ["1-3", "1 to 3", "class 1", "class 2", "class 3", "primary", "lower primary", 
                    "young children", "younger kids", "small kids", "little ones", "ages 6-8", "6-8 years"],
            "4-5": ["4-5", "4 to 5", "class 4", "class 5", "upper primary", "ages 9-11", "9-11 years"],
            "6-8": ["6-8", "6 to 8", "class 6", "class 7", "class 8", "middle school", "6th", "7th", "8th",
                    "pre-teens", "preteens", "ages 11-14", "11-14 years", "grade 6", "grade 7", "grade 8"],
            "9-10": ["9-10", "9 to 10", "class 9", "class 10", "9th", "10th", "high school", "secondary", "higher secondary",
                     "teenagers", "teens", "ages 14-16", "14-16 years", "grade 9", "grade 10"],
            "11-12": ["11-12", "11 to 12", "class 11", "class 12", "11th", "12th", "senior", 
                      "ages 16-18", "16-18 years", "college prep", "grade 11", "grade 12"]
        }
        
        for grade in catalog.get("grades", []):
            patterns = grade_patterns.get(grade, [grade])
            if any(pattern in txt for pattern in patterns):
                if grade not in grades:  # Avoid duplicates
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
    
    # Extract language/medium
    language_medium = None
    if "hindi" in txt or "हिंदी" in txt or "comfortable in hindi" in txt:
        language_medium = "Hindi"
    elif "tamil" in txt or "தமிழ்" in txt or "comfortable in tamil" in txt:
        language_medium = "Tamil"
    elif "kannada" in txt or "ಕನ್ನಡ" in txt:
        language_medium = "Kannada"
    elif "telugu" in txt or "తెలుగు" in txt:
        language_medium = "Telugu"
    elif "english" in txt:
        # Only set if explicitly mentioned as medium (not as subject)
        if "in english" in txt or "english medium" in txt or "comfortable in english" in txt:
            language_medium = "English"
    elif "both" in txt or "bilingual" in txt or "any language" in txt:
        language_medium = "Both"
    
    # Calculate confidence based on what was extracted
    confidence = 0.0
    if subjects:
        confidence += 0.4
    if grades:
        confidence += 0.3
    if availability:
        confidence += 0.2
    if language_medium:
        confidence += 0.1
    
    return subjects, grades, availability, language_medium, confidence

# ------------- LLM ENHANCEMENT -------------

async def _llm_enhance(
    text: str,
    catalog: Dict[str, List[str]],
    fast_results: tuple[List[str], List[str], Optional[str], Optional[str], float]
) -> ExtractOutput:
    """
    Use LLM to enhance or validate fast extraction results.
    """
    subjects, grades, availability, language_medium, _ = fast_results
    
    prompt = f"""You are extracting teacher profile information. Analyze the text carefully and match to the provided catalog.

USER TEXT: "{text}"

AVAILABLE OPTIONS:
- Subjects: {', '.join(catalog.get("subjects", []))}
- Grades: {', '.join(catalog.get("grades", []))}

EXTRACTION RULES:
1. Subject Matching:
   - "numbers", "arithmetic", "calculation" = Math
   - "language", "grammar", "writing", "reading" = English (as subject, not medium)
   - "natural science", "experiments" = Science
   - Match ONLY from catalog: {', '.join(catalog.get("subjects", []))}
   
2. Grade Matching:
   - "any grade", "all grades", "flexible" = ALL grades from catalog
   - "young children", "younger kids", "small kids" = 1-3
   - "grade 6", "grade 7", "grade 8", "middle school" = 6-8
   - "teenagers", "high school" = 9-10
   - Match ONLY from catalog: {', '.join(catalog.get("grades", []))}
   
3. Language/Medium of Instruction:
   - "in Hindi", "Hindi medium" = "Hindi"
   - "in Tamil", "Tamil medium" = "Tamil"
   - "in English" = "English"
   - "both", "bilingual" = "Both"
   - Note: "English" as subject is different from "in English" as medium
   
4. Confidence:
   - 0.9-1.0 if very clear and specific
   - 0.7-0.9 if correctly inferred
   - 0.5-0.7 if somewhat ambiguous

FAST EXTRACTION (validate and improve):
- Subjects: {subjects if subjects else "none"}
- Grades: {grades if grades else "none"}
- Language: {language_medium if language_medium else "not mentioned"}
- Availability: {availability if availability else "none"}

YOUR TASK:
Validate and enhance. Return ONLY this JSON (no markdown, no explanation):
{{
  "subjects": ["subjects from catalog"],
  "grades": ["grades from catalog - if 'any grade' mentioned, include ALL"],
  "availability_note": "pattern if mentioned",
  "language_medium": "Hindi/Tamil/English/Both or null",
  "timezone": null,
  "confidence": 0.85,
  "notes": ["key inferences: numbers->Math, younger kids->1-3, in tamil->Tamil"]
}}"""

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
            language_medium=language_medium,
            confidence=fast_results[4],
            extraction_method="rule_based_only"
        )
    
    # Merge LLM results with fast results (LLM takes precedence)
    return ExtractOutput(
        subjects=data.get("subjects", subjects),
        grades=data.get("grades", grades),
        availability_note=data.get("availability_note", availability),
        language_medium=data.get("language_medium", language_medium),
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
    subjects, grades, availability, language_medium, confidence = fast_results
    
    print(f"[llm.extract_profile_fields] Fast extraction: subjects={subjects}, grades={grades}, language={language_medium}, confidence={confidence}")
    
    # If LLM disabled or decent confidence, return fast results
    # Threshold: 0.4 is good enough if we found something concrete
    if not body.use_llm or confidence >= 0.4 or (subjects or grades):
        return ExtractOutput(
            subjects=subjects,
            grades=grades,
            availability_note=availability,
            language_medium=language_medium,
            confidence=max(confidence, 0.7) if (subjects or grades) else confidence,
            extraction_method="rule_based",
            extracted_something=bool(subjects or grades or availability)
        )
    
    # Stage 2: LLM enhancement only for truly ambiguous cases (nothing extracted)
    print(f"[llm.extract_profile_fields] No clear extraction (confidence={confidence}), calling LLM for enhancement")
    enhanced = await _llm_enhance(body.text, body.catalog, fast_results)
    
    print(f"[llm.extract_profile_fields] LLM enhanced: subjects={enhanced.subjects}, confidence={enhanced.confidence}")
    
    enhanced.extracted_something = bool(enhanced.subjects or enhanced.grades or enhanced.availability_note)
    
    # Ensure language_medium is set (backward compatibility)
    if not hasattr(enhanced, 'language_medium') or enhanced.language_medium is None:
        enhanced.language_medium = None
    
    return enhanced
