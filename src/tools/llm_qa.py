"""
LLM QA Tool
- Generate FAQ answers using LLM with RAG context from knowledge.search
- Provides cleaner abstraction than generic llm.call for QA tasks
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from tools.llm_core import call_llm_for_text

router = APIRouter()

# --------- Models ---------

class Snippet(BaseModel):
    id: str
    title: str
    text: str

class UserProfile(BaseModel):
    name: Optional[str] = None
    tz: Optional[str] = None

class LLMQARequest(BaseModel):
    question: str = Field(..., description="User's question")
    snippets: List[Snippet] = Field(..., description="RAG context from knowledge.search")
    policy_version: Optional[str] = Field(None, description="Optional policy version")
    knowledge_version: Optional[str] = Field(None, description="Optional knowledge version")
    user_profile: Optional[UserProfile] = Field(None, description="Optional user context")

class LLMQAResponse(BaseModel):
    answer: str = Field(..., description="Generated FAQ answer")
    snippet_ids_used: List[str] = Field(default_factory=list, description="Which snippets influenced the answer")

# --------- Answer Generation ---------

async def _generate_qa_answer(
    question: str,
    snippets: List[Snippet],
    policy_version: Optional[str],
    knowledge_version: Optional[str],
    user_profile: Optional[UserProfile]
) -> tuple[str, List[str]]:
    """
    Generate FAQ answer using LLM with RAG context.
    
    Returns:
        (answer_text, snippet_ids_used)
    """
    # Build context from snippets
    snippet_texts = []
    snippet_ids = []
    for snippet in snippets:
        snippet_texts.append(f"- {snippet.title}: {snippet.text}")
        snippet_ids.append(snippet.id)
    
    context_text = "\n".join(snippet_texts) if snippet_texts else "(no relevant snippets found)"
    
    # Build user context
    user_context_parts = []
    if user_profile:
        if user_profile.name:
            user_context_parts.append(f"Volunteer name: {user_profile.name}")
        if user_profile.tz:
            user_context_parts.append(f"Timezone: {user_profile.tz}")
    user_context = "\n".join(user_context_parts) if user_context_parts else "No user context provided"
    
    # Build system prompt (per spec)
    system_prompt = """You are Sia, SERVE's onboarding assistant. Answer the volunteer's question in 2–4 short lines.

Use only the provided snippets/policy. Do NOT promise payment. 

If unsure, say so briefly and suggest asking the coordinator in orientation.

End with: "Shall we schedule your orientation?"

Return plain text (no JSON, no markdown)."""
    
    # Build user prompt (per spec)
    context_dict = {}
    if policy_version:
        context_dict["policy_version"] = policy_version
    if knowledge_version:
        context_dict["knowledge_version"] = knowledge_version
    
    context_str = "\n".join([f"{k}: {v}" for k, v in context_dict.items()]) if context_dict else "No version info"
    
    user_prompt = f"""Context:
{{
  "policy_version": "{policy_version or 'not specified'}",
  "knowledge_version": "{knowledge_version or 'not specified'}",
  "snippets": [
{context_text}
  ],
  "user_profile": {{{user_context}}}
}}

User question: {question}"""
    
    # Call LLM
    full_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
    
    answer_text, error = await call_llm_for_text(
        prompt=full_prompt,
        temperature=0.2,
        max_tokens=200
    )
    
    # Fallback if LLM fails
    if error or not answer_text:
        # Use first snippet as fallback
        if snippets:
            fallback_text = snippets[0].text
            answer_text = f"{fallback_text}\n\nShall we schedule your orientation?"
        else:
            answer_text = "I'm not sure about that. Would you like to ask the coordinator during orientation?\n\nShall we schedule your orientation?"
    
    return answer_text.strip(), snippet_ids

# --------- Endpoint ---------

@router.post("/llm.qa", response_model=LLMQAResponse)
async def llm_qa(req: LLMQARequest) -> LLMQAResponse:
    """
    Generate FAQ answer using LLM with RAG context.
    
    This tool provides a cleaner abstraction than generic llm.call for QA tasks.
    It takes snippets from knowledge.search and generates a concise, policy-safe answer.
    
    The answer will always end with "Shall we schedule your orientation?" as per spec.
    """
    answer, snippet_ids = await _generate_qa_answer(
        question=req.question,
        snippets=req.snippets,
        policy_version=req.policy_version,
        knowledge_version=req.knowledge_version,
        user_profile=req.user_profile
    )
    
    return LLMQAResponse(
        answer=answer,
        snippet_ids_used=snippet_ids
    )

