"""
Knowledge Search Tool
- Search knowledge base for FAQ snippets relevant to user queries
- Supports semantic/keyword search and policy version filtering
- Returns structured snippets with relevance scores
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import re

router = APIRouter()

# --------- Models ---------

class KnowledgeSnippet(BaseModel):
    id: str
    title: str
    text: str
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score (0-1)")

class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., description="User's question or search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of snippets to return")
    policy_version: Optional[str] = Field(None, description="Optional: policy version for filtering (e.g., 'v1.2')")

class KnowledgeSearchResponse(BaseModel):
    snippets: List[KnowledgeSnippet]
    knowledge_version: str = Field(default="v2.1", description="Version of knowledge base used")

# --------- Knowledge Base ---------

# Structured knowledge base with FAQ snippets
# In production, this would be stored in a vector database or search index
_KNOWLEDGE_BASE = {
    "knowledge_version": "v2.1",
    "snippets": [
        {
            "id": "faq_serve_intro",
            "title": "What is SERVE?",
            "text": "SERVE helps thousands of children learn English, Science, and Maths through volunteers like you. You teach online — they learn in school — and our local coordinators make sure everything runs smoothly.",
            "tags": ["about_serve", "organization"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_time_commitment",
            "title": "Time Commitment",
            "text": "You'll teach live online while students sit in their school smart classroom. Usually ~2 hours/week.",
            "tags": ["time_process", "hours"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_training",
            "title": "Training & Support",
            "text": "Yes! You'll get the pedagogy from the volunteer coordinator after you get assigned, and a local coordinator supports you during classes.",
            "tags": ["support", "training"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_certificate",
            "title": "Certificate",
            "text": "We provide a volunteer certificate after you complete the required sessions as per policy.",
            "tags": ["certificate", "document"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_subjects",
            "title": "Subjects & Grades",
            "text": "Most volunteers teach English, Math or Science for grades 5–8 (varies by school). We'll align your preferences during scheduling.",
            "tags": ["subjects_grades", "teaching"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_tech",
            "title": "Tech Requirements",
            "text": "A phone or laptop with stable internet is enough. We'll share the Meet link for sessions.",
            "tags": ["tech", "device"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_orientation",
            "title": "Orientation Process",
            "text": "There will not be any separate orientation session. If you need more information, you can ask me or a volunteer coordinator will get in touch with you",
            "tags": ["orientation", "onboarding", "training"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_scheduling",
            "title": "Scheduling & Availability",
            "text": "Live classes happen only on weekdays between 8:00 and 15:00 (school hours). You'll schedule your sessions based on your availability.",
            "tags": ["scheduling", "availability", "time"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_compensation",
            "title": "Compensation",
            "text": "This is a 100% volunteer role; there is no pay. We provide a volunteer certificate after completion.",
            "tags": ["compensation", "pay", "volunteer"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_commitment_duration",
            "title": "Commitment Duration",
            "text": "We ask for at least 3 months of commitment. Typical commitment is ~2 hours per week, split across different weekdays.",
            "tags": ["commitment", "duration", "months"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_subject_choice",
            "title": "Can I teach subjects other than English (like Maths/Science)?",
            "text": "Yes. Most roles are in English, Maths, or Science for grades 5–8. Subject availability may vary by school/region; we’ll match your interest during scheduling.",
            "tags": ["subjects", "math", "science", "grades", "availability"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_weekends",
            "title": "Can I teach during weekends?",
            "text": "Typically no. Classes run on school days and hours. In rare cases (remedials/events) a weekend slot may open regionally, but it’s not the norm.",
            "tags": ["weekends", "policy", "schedule"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_same_day_two_hours",
            "title": "Can I take both the sessions on Saturday?",
            "text": "No. The ~2 hours/week are meant to be split across different weekdays during school hours. Same‑day 2 hours isn’t part of the standard model.",
            "tags": ["same_day", "hours", "policy", "schedule"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_mobile_ok",
            "title": "Can I teach using my mobile phone?",
            "text": "No. tablet or a laptop only are acceptable. A quiet space and a basic headset help with clarity.",
            "tags": ["device", "mobile", "internet", "requirements"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_certificate_policy",
            "title": "Will I get a volunteering certificate?",
            "text": "Yes. This is a 100% volunteer role (no pay). We issue a volunteer certificate after you complete the required sessions per policy.",
            "tags": ["certificate", "volunteer", "policy"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_commitment_2months",
            "title": "I can commit for 2 months—can I still get involved?",
            "text": "You’re welcome to start. We request ~3 months to stabilize learning for students. If your time frees up later, we can extend; if not, we’ll make the most of your 2 months.",
            "tags": ["commitment", "duration", "flexibility"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_materials",
            "title": "Will I get curriculum and study material?",
            "text": "Yes. We share a simple curriculum outline, lesson resources, and best‑practice guides. Orientation covers how to use them effectively.",
            "tags": ["curriculum", "materials", "resources", "orientation"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_higher_grades",
            "title": "Are there opportunities to teach higher grades (8+)?",
            "text": "Often yes, but school demand varies. We’ll note your preference and try to match it; if not available, we’ll suggest nearby grades.",
            "tags": ["grades", "opportunities", "placement"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_language_expectation",
            "title": "If the medium is Hindi, do I have to translate every line?",
            "text": "No. Teach primarily in simple English with supportive Hindi phrases where needed. Keep instructions clear; we don’t expect literal translation of every line.",
            "tags": ["language", "hindi", "teaching_style"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_under_18",
            "title": "I’m under 18 but eager to help—can I teach?",
            "text": "Teaching roles require you to be 18+. If you’re under 18, we can explore non‑teaching contributions or have you reapply once you turn 18.",
            "tags": ["age", "eligibility", "policy"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_curriculum_3months",
            "title": "Do I have to complete all curriculum in 3 months? Will lesson plans be shared?",
            "text": "No. The 3‑month window is for continuity, not for finishing the entire curriculum. We provide a simple scope/sequence and lesson plans to pace sessions sensibly.",
            "tags": ["curriculum", "lesson_plan", "timeline"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_start_later",
            "title": "I can’t start immediately. Will the opportunity be available in ~2 months?",
            "text": "Yes. You can defer your start. We’ll note your timeline and reconnect closer to your availability to match you with an appropriate school.",
            "tags": ["deferral", "availability", "timeline"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_whatsapp_alt",
            "title": "WhatsApp doesn’t work in my region. How do I communicate with the school?",
            "text": "We can use alternative channels (email/Meet/telephony) coordinated via the local team. Orientation will confirm your preferred channel and share the process.",
            "tags": ["communication", "whatsapp", "region", "alternate"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_weekday_plus_sat",
            "title": "Can I teach one weekday plus Saturday 12–1 PM?",
            "text": "Saturday classes are uncommon and depend on regional arrangements. We’ll try to match your weekday; Saturday afternoon is not guaranteed.",
            "tags": ["weekends", "availability", "matching"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_mixed_levels",
            "title": "If students have different levels in one class, how do I handle it?",
            "text": "Use tiered questions, quick checks, and small practice tasks. We share level‑wise activities and tips in orientation and the teacher guide.",
            "tags": ["pedagogy", "differentiation", "classroom"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_homework_tests",
            "title": "Can I give homework? Can I conduct tests?",
            "text": "Short practice tasks are fine. Formal tests are coordinated with the school. Keep any assessments brief and share outcomes with the coordinator.",
            "tags": ["homework", "assessment", "school_policy"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_reschedule",
            "title": "If I can’t take a session once in a while, can I reschedule?",
            "text": "Yes, with prior notice. Inform the coordinator early so the school can adjust. We’ll aim to find a nearby slot within the week.",
            "tags": ["reschedule", "attendance", "coordination"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_multiple_schools",
            "title": "Can I teach in more than one school?",
            "text": "Possibly, based on demand and your schedule. We recommend starting with one class and adding another once your rhythm is steady.",
            "tags": ["capacity", "multi_school", "availability"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_no_experience",
            "title": "I don’t have prior teaching experience. Can I still teach?",
            "text": "Yes. Orientation covers the basics, and we share simple lesson flows and materials. Enthusiasm and consistency matter most.",
            "tags": ["experience", "orientation", "support"],
            "policy_version": "v1.2"
        },
        {
            "id": "faq_late_join_extension",
            "title": "If the school joins 20 minutes late, should I extend by 20 minutes?",
            "text": "Not necessarily. Follow the agreed school window. If it’s feasible and the classroom is available, a brief extension can be coordinated with the coordinator.",
            "tags": ["timing", "session_length", "coordination"],
            "policy_version": "v1.2"
        }
    ]
}

# --------- Search Functions ---------

def _tokenize(text: str) -> set:
    """Tokenize text into lowercase words"""
    return set(re.findall(r"\w+", text.lower()))

def _score_relevance(query: str, snippet: Dict) -> float:
    """
    Calculate relevance score between query and snippet.
    Uses keyword matching on title, text, and tags.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    
    # Score title (higher weight)
    title_tokens = _tokenize(snippet.get("title", ""))
    title_score = len(query_tokens & title_tokens) / max(len(query_tokens), 1) * 1.5
    
    # Score text
    text_tokens = _tokenize(snippet.get("text", ""))
    text_score = len(query_tokens & text_tokens) / max(len(query_tokens), 1) * 1.0
    
    # Score tags (higher weight)
    tags = snippet.get("tags", [])
    tag_tokens = set()
    for tag in tags:
        tag_tokens.update(_tokenize(tag))
    tag_score = len(query_tokens & tag_tokens) / max(len(query_tokens), 1) * 1.2
    
    # Combine scores (normalize to 0-1)
    total_score = (title_score + text_score + tag_score) / 3.7  # Normalize by max possible weight
    return min(max(total_score, 0.0), 1.0)

def _filter_by_policy_version(snippets: List[Dict], policy_version: Optional[str]) -> List[Dict]:
    """Filter snippets by policy version if provided"""
    if not policy_version:
        return snippets
    
    # Return snippets that match the policy version or are compatible
    # For now, exact match; can be extended with version compatibility logic
    filtered = []
    for snippet in snippets:
        snippet_version = snippet.get("policy_version")
        if snippet_version == policy_version:
            filtered.append(snippet)
        # If no policy_version specified in snippet, include it (backward compatibility)
        elif not snippet_version:
            filtered.append(snippet)
    
    return filtered

# --------- Endpoint ---------

@router.post("/knowledge.search", response_model=KnowledgeSearchResponse)
async def knowledge_search(req: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """
    Search knowledge base for FAQ snippets relevant to a user query.
    
    Performs semantic/keyword search over FAQ content and returns top-k most relevant snippets.
    Supports policy version filtering if provided.
    
    Returns empty array if no matches or on error (client should handle gracefully).
    """
    try:
        # Get all snippets
        all_snippets = _KNOWLEDGE_BASE.get("snippets", [])
        
        # Filter by policy version if provided
        if req.policy_version:
            all_snippets = _filter_by_policy_version(all_snippets, req.policy_version)
        
        # Score and rank snippets
        scored = []
        for snippet in all_snippets:
            score = _score_relevance(req.query, snippet)
            if score > 0:
                scored.append((snippet, score))
        
        # Sort by relevance (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k
        top_snippets = scored[:req.top_k]
        
        # Convert to response format
        result_snippets = [
            KnowledgeSnippet(
                id=snippet["id"],
                title=snippet["title"],
                text=snippet["text"],
                relevance_score=score
            )
            for snippet, score in top_snippets
        ]
        
        return KnowledgeSearchResponse(
            snippets=result_snippets,
            knowledge_version=_KNOWLEDGE_BASE.get("knowledge_version", "v2.1")
        )
        
    except Exception as e:
        # Return empty array on error (per spec)
        print(f"[knowledge.search] Error: {str(e)}")
        return KnowledgeSearchResponse(
            snippets=[],
            knowledge_version=_KNOWLEDGE_BASE.get("knowledge_version", "v2.1")
        )

