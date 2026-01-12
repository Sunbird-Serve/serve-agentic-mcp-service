from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from tools.wa_send import SendMessageInput, publish_wa_out
from tools import llm_extract, llm_reply, calendar_create
from tools import time_parse, llm_time_parse, llm_core, time_refine, intent_detect, llm_classify, llm_smart_edit
from tools import faq_answer
from tools import onboarding_parse
from tools import consent, eligibility, preferences, slots, reminders, telemetry, profile
from tools import policy
from tools import prefs_polish, intent_orientation
from tools import deferral, state
from tools import onboarding_next
from tools import llm_humanize
from tools import knowledge, llm_qa, onboarding_turns
from tools import serve_needs, serve_fulfill
from tools import wa_video

app = FastAPI(title="serve-agentic-mcp-service", version="0.1.0")

# ---- Health & introspection ----
@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/mcp/tools")
async def list_tools():
    return {
        "tools": [
            {"name": "wa.send_message", "input": {"to": "str", "text": "str"}},
            {"name": "llm.call", "input": {"messages": "list", "temperature": "float", "max_tokens": "int"}, "description": "Generic LLM interface"},
            {"name": "intent.detect", "input": {"text": "str", "context": "str"}, "description": "Detect user intent (confirmation/refinement/rejection)"},
            {"name": "llm.extract_profile_fields"},
            {"name": "llm.generate_reply"},
            {"name": "llm.parse_time", "input": {"text": "str", "tz": "str", "duration_minutes": "int"}, "description": "Parse natural language time"},
            {"name": "time.parse_options", "input": {"text": "str", "tz": "str", "duration_minutes": "int"}, "description": "Smart time parser (fast + LLM)"},
            {"name": "time.refine_slots", "input": {"original_slots": "list", "refinement_text": "str"}, "description": "Refine/modify time slots"},
            {"name": "llm.handle_smart_edit", "input": {"conversation_history": "list", "current_profile": "object", "user_input": "str"}, "description": "Smart profile editing"},
            {"name": "llm.humanize_weekday_confirmation", "input": {"flow_state_summary": "str", "user_input": "str", "locale": "str"}, "description": "Human layer: weekday confirmation reply"},
            {"name": "calendar.create_event"}
        ]
    }

# ---- MCP Tools (HTTP style) ----
@app.post("/mcp/wa.send_message")
async def wa_send_message(payload: SendMessageInput):
    try:
        result = await publish_wa_out(payload.to, payload.text, payload.buttons)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to send message: {e!r}")

app.include_router(llm_core.router, prefix="/mcp")  # Generic LLM interface
app.include_router(intent_detect.router, prefix="/mcp")  # Intent detection (prevents loops)
app.include_router(llm_extract.router, prefix="/mcp")
app.include_router(llm_reply.router, prefix="/mcp")
app.include_router(llm_classify.router, prefix="/mcp")  # Response classification for eligibility
app.include_router(llm_smart_edit.router, prefix="/mcp")  # Smart profile editing
app.include_router(llm_time_parse.router, prefix="/mcp")  # Specialized LLM tool
app.include_router(calendar_create.router, prefix="/mcp")
app.include_router(time_parse.router, prefix="/mcp")  # Orchestrator tool
app.include_router(time_refine.router, prefix="/mcp")  # Refinement tool
app.include_router(llm_humanize.router, prefix="/mcp")  # Human layer tool
app.include_router(faq_answer.router, prefix="/mcp")  # FAQ tool
app.include_router(onboarding_parse.router, prefix="/mcp")  # Onboarding parser
app.include_router(consent.router, prefix="/mcp")
app.include_router(eligibility.router, prefix="/mcp")
app.include_router(preferences.router, prefix="/mcp")
app.include_router(slots.router, prefix="/mcp")
app.include_router(reminders.router, prefix="/mcp")
app.include_router(telemetry.router, prefix="/mcp")
app.include_router(profile.router, prefix="/mcp")
app.include_router(policy.router, prefix="/mcp")
app.include_router(prefs_polish.router, prefix="/mcp")
app.include_router(intent_orientation.router, prefix="/mcp")
app.include_router(deferral.router, prefix="/mcp")
app.include_router(state.router, prefix="/mcp")
app.include_router(onboarding_next.router, prefix="/mcp")
app.include_router(knowledge.router, prefix="/mcp")  # Knowledge search
app.include_router(llm_qa.router, prefix="/mcp")  # LLM QA tool
app.include_router(onboarding_turns.router, prefix="/mcp")  # Unified onboarding turn handler
app.include_router(serve_needs.router, prefix="/mcp")  # Serve needs list
app.include_router(serve_fulfill.router, prefix="/mcp")  # Serve fulfillment nomination
app.include_router(wa_video.router, prefix="/mcp")  # WhatsApp video sending

# Mount MCP JSON-RPC server AFTER all routes are registered
# This way it doesn't override existing endpoints
from mcp_server import mcp_app
app.mount("/mcp/v1", mcp_app)