"""
Orientation vs Class Clarifier
- Classifies if a user's question refers to orientation or class, or is unclear
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from tools.llm_core import call_llm_for_json

router = APIRouter()

class ClarifyRequest(BaseModel):
    text: str
    state: Optional[str] = Field(default=None, description="Current onboarding state")

class ClarifyResponse(BaseModel):
    intent: str  # ORIENTATION | CLASS | UNCLEAR
    confidence: float

@router.post("/intent.classify_orientation", response_model=ClarifyResponse)
async def classify_orientation(req: ClarifyRequest) -> ClarifyResponse:
    system = (
        "You are a precise classifier for SERVE onboarding. Classify if the user's message is about ORIENTATION (onboarding call) "
        "or CLASS (actual teaching session/commitment), or UNCLEAR. Return JSON only."
    )
    user = (
        f"state={req.state or 'UNKNOWN'}\n"
        f"text=\"{req.text}\"\n"
        "Output JSON: {\"intent\": \"ORIENTATION|CLASS|UNCLEAR\", \"confidence\": 0.0}"
    )
    prompt = f"System:\n{system}\n\nUser:\n{user}"
    data, err = await call_llm_for_json(prompt=prompt, temperature=0.0, max_tokens=60)
    if err or not isinstance(data, dict):
        return ClarifyResponse(intent="UNCLEAR", confidence=0.4)
    intent = str(data.get("intent", "UNCLEAR")).upper()
    if intent not in ("ORIENTATION","CLASS","UNCLEAR"):
        intent = "UNCLEAR"
    conf = float(data.get("confidence", 0.5))
    return ClarifyResponse(intent=intent, confidence=conf)


