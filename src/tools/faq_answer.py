"""
FAQ Answer Tool
- Retrieves KB snippets and composes a concise, policy-safe answer
- Falls back to KB-only if LLM unavailable
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import re

from tools.llm_core import call_llm_for_text

router = APIRouter()

# --------- Models ---------

class Source(BaseModel):
    id: str
    confidence: float = Field(ge=0.0, le=1.0)

class FAQRequest(BaseModel):
    question: str
    policy_context: str
    kb_scope: str = Field(default="onboarding-basic")
    top_k: int = Field(default=3, ge=1, le=10)
    state: Optional[str] = Field(default=None, description="Optional: current onboarding state for routing")

class FAQResponse(BaseModel):
    answer: str = ""
    bridge: str = ""
    sources: Optional[List[Source]] = None
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    action: Optional[str] = Field(default="answer", description="answer | defer_to_step")
    defer_to: Optional[str] = Field(default=None, description="target step when action=defer_to_step")

# --------- Minimal KB ---------
# In real deployment, wire to a vector store or search index
_KB: Dict[str, Dict[str, str]] = {
    "onboarding-basic": {
        "timings": "Live classes happen only on weekdays between 8:00 and 15:00 (school hours).",
        "orientation": "Orientation is a brief onboarding call to share the flow, best practices, and answer questions.",
        "devices": "A smartphone or laptop with stable internet is sufficient. A quiet space and headset help.",
        "commitment": "Typical commitment is ~2 hours per week; short 20–30 minute sessions are welcome.",
        "compensation": "This is a volunteer role; there is no pay.",
    }
}

# --------- Retriever ---------

def _score(query: str, text: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    t = set(re.findall(r"\w+", text.lower()))
    if not q or not t:
        return 0.0
    overlap = len(q & t)
    return overlap / max(len(q), 1)

def retrieve_snippets(scope: str, query: str, top_k: int) -> List[Source]:
    kb = _KB.get(scope, {})
    scored = []
    for sid, snippet in kb.items():
        s = _score(query, snippet)
        if s > 0:
            scored.append(Source(id=sid, confidence=min(max(s, 0.1), 1.0)))
    scored.sort(key=lambda x: x.confidence, reverse=True)
    return scored[:top_k]

# --------- Composer ---------

async def _compose_answer(policy: str, question: str, snippets: List[Source], scope: str, state: Optional[str]) -> FAQResponse:
    # State-aware routing: defer commitment-policy queries back to step logic
    q_low = question.lower()
    if (state or "").upper() == "ELIGIBILITY_PART2":
        if any(kw in q_low for kw in ["same day", "same-day", "same day?", "2 hrs same day", "2 hours same day", "two hours same day"]):
            return FAQResponse(action="defer_to_step", defer_to="commitment_policy", confidence=0.7)
    kb_texts = []
    for s in snippets:
        kb_texts.append(_KB.get(scope, {}).get(s.id, ""))
    kb_concat = " \n".join([t for t in kb_texts if t])

    system = (
        "You are a concise FAQ assistant for a volunteer onboarding flow. "
        "Answer briefly (<= 2 sentences), policy-safe, and end with ONE short bridge question to return to the current step. "
        "Output plain text only."
    )
    user = (
        f"Policy context: {policy}\n\n"
        f"KB snippets:\n{kb_concat if kb_concat else '(none)'}\n\n"
        f"Question: {question}\n\n"
        f"Write 1-2 sentence answer, then 1 short question to continue."
    )

    text, error = await call_llm_for_text(
        prompt=f"System:\n{system}\n\nUser:\n{user}",
        temperature=0.2,
        max_tokens=140
    )

    # Fallback: KB-only
    if error or not text:
        answer = (kb_concat or "Let me share the basics.").strip()
        # Trim to <= 2 sentences
        sents = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
        if not sents:
            answer = "Here is a quick summary."
        else:
            answer = (sents[0] + ('. ' + sents[1] + '.') if len(sents) > 1 else '.')
        bridge = "Shall we continue with the current step?"
        conf = max((snippets[0].confidence if snippets else 0.4), 0.4)
        return FAQResponse(answer=answer, bridge=bridge, sources=snippets or None, confidence=conf)

    # Split answer and bridge; prefer taking the last question as bridge and the rest as answer
    lines = text.strip().splitlines()
    merged = " ".join([ln.strip() for ln in lines if ln.strip()])
    if '?' in merged:
        parts_q = merged.split('?')
        bridge_candidate = parts_q[-2].strip() + '?' if len(parts_q) >= 2 else merged.strip()
        # Everything before the last question mark becomes the answer component
        answer_candidate = '?'.join(parts_q[:-1]).rstrip('?').strip()
        answer_text = answer_candidate.strip()
        bridge = bridge_candidate.strip()
    else:
        # No explicit question; synthesize a bridge
        answer_text = merged
        bridge = "Does that help, and shall we proceed with the next step?"

    # Enforce <= 2 sentences for answer
    sents = [s.strip() for s in re.split(r"[.!?]", answer_text) if s.strip()]
    if len(sents) > 2:
        answer_text = sents[0] + '. ' + sents[1] + '.'
    elif sents and not answer_text.endswith('.'):
        answer_text = answer_text + '.'

    # Ensure bridge is exactly one short question
    if '?' in bridge:
        bridge = bridge.split('?')[0].strip() + '?'
    else:
        bridge = "Shall we continue with the current step?"

    # Confidence: boost if input looks like a question and we had KB support
    base_conf = max([s.confidence for s in snippets], default=0.5)
    is_question = '?' in question or question.lower().startswith(('what','when','how','why','can','do','does','is','are'))
    conf = min(1.0, (base_conf + (0.2 if is_question else -0.1)))
    # De-dup identical lines and near-duplicates differing only by trailing punctuation
    def _norm(s: str) -> str:
        return re.sub(r'[\s]+', ' ', s.strip().rstrip('.!?')).lower()
    seen = set()
    ordered: list[str] = []
    for part in [answer_text, bridge]:
        if not part:
            continue
        key = _norm(part)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(part)
    answer_out = ordered[0] if ordered else ""
    bridge_out = ordered[1] if len(ordered) > 1 else ""

    return FAQResponse(answer=answer_out, bridge=bridge_out, sources=snippets or None, confidence=max(0.0, conf), action="answer")

# --------- Endpoint ---------

@router.post("/faq.answer", response_model=FAQResponse)
async def faq_answer(req: FAQRequest) -> FAQResponse:
    # Deferral routing: if user is effectively deferring in ELIGIBILITY_PART2, defer to step logic
    q_low = req.question.lower()
    if (req.state or "").upper() == "ELIGIBILITY_PART2":
        if any(kw in q_low for kw in ["think and get back", "get back later", "later", "not now", "not yet", "remind me", "reminder"]):
            return FAQResponse(action="defer_to_step", defer_to="deferral", confidence=0.8)

    snippets = retrieve_snippets(req.kb_scope, req.question, req.top_k)
    return await _compose_answer(req.policy_context, req.question, snippets, req.kb_scope, req.state)
