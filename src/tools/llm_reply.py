from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

class ReplyInput(BaseModel):
    purpose: str  # "ask_next_question" | "confirm" | "fallback"
    state: str
    profile_partial: dict = {}
    locale: str = "en-IN"

class ReplyOutput(BaseModel):
    text: str

@router.post("/llm.generate_reply", response_model=ReplyOutput)
async def generate_reply(body: ReplyInput):
    #  for now, canned prompts based on state
    if body.purpose == "ask_next_question" and body.state == "ASK_GRADES":
        return ReplyOutput(text="Great! Which grades or age groups would you like to teach? (e.g., 6–8)")
    if body.purpose == "confirm" and body.state == "DONE":
        s = ", ".join(body.profile_partial.get("subjects", [])) or "N/A"
        g = body.profile_partial.get("grades", "N/A")
        return ReplyOutput(text=f"Thanks! ✅\nSubjects: {s}\nGrades: {g}\nWe’ll follow up soon.")
    return ReplyOutput(text="Got it. Could you please clarify?")
