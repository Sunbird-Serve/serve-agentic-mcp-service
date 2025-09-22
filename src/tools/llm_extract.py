from pydantic import BaseModel, Field
from fastapi import APIRouter
from typing import List, Dict

router = APIRouter()

class ExtractInput(BaseModel):
    state: str
    text: str
    locale: str = "en-IN"
    catalog: Dict[str, List[str]] = Field(default_factory=dict)

class ExtractOutput(BaseModel):
    subjects: List[str] = []
    grades: List[str] = []
    availability_note: str | None = None
    timezone: str | None = None
    confidence: float = 0.0
    notes: List[str] = []

@router.post("/llm.extract_profile_fields", response_model=ExtractOutput)
async def extract_profile_fields(body: ExtractInput):
    #  a tiny rule-based stub:
    txt = body.text.lower()
    subs = [s for s in body.catalog.get("subjects", []) if s.lower() in txt]
    grs = [g for g in body.catalog.get("grades", []) if g in body.text]
    avail = "weekend" if "weekend" in txt else ( "weekday evenings" if "evening" in txt else None )
    return ExtractOutput(subjects=subs, grades=grs, availability_note=avail, confidence=0.5)
