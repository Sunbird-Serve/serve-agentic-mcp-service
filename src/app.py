from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from tools.wa_send import SendMessageInput, publish_wa_out
from tools import llm_extract, llm_reply, calendar_create
from tools import time_parse, llm_time_parse, llm_core, time_refine, intent_detect

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
            {"name": "llm.call", "input": {"prompt": "str", "format": "str", "temperature": "float"}, "description": "Generic LLM interface"},
            {"name": "intent.detect", "input": {"text": "str", "context": "str"}, "description": "Detect user intent (confirmation/refinement/rejection)"},
            {"name": "llm.extract_profile_fields"},
            {"name": "llm.generate_reply"},
            {"name": "llm.parse_time", "input": {"text": "str", "tz": "str", "duration_minutes": "int"}, "description": "Parse natural language time"},
            {"name": "time.parse_options", "input": {"text": "str", "tz": "str", "duration_minutes": "int"}, "description": "Smart time parser (fast + LLM)"},
            {"name": "time.refine_slots", "input": {"original_slots": "list", "refinement_text": "str"}, "description": "Refine/modify time slots"},
            {"name": "calendar.create_event"}
        ]
    }

# ---- MCP Tools (HTTP style) ----
@app.post("/mcp/wa.send_message")
async def wa_send_message(payload: SendMessageInput):
    try:
        result = await publish_wa_out(payload.to, payload.text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to send message: {e!r}")

app.include_router(llm_core.router, prefix="/mcp")  # Generic LLM interface
app.include_router(intent_detect.router, prefix="/mcp")  # Intent detection (prevents loops)
app.include_router(llm_extract.router, prefix="/mcp")
app.include_router(llm_reply.router, prefix="/mcp")
app.include_router(llm_time_parse.router, prefix="/mcp")  # Specialized LLM tool
app.include_router(calendar_create.router, prefix="/mcp")
app.include_router(time_parse.router, prefix="/mcp")  # Orchestrator tool
app.include_router(time_refine.router, prefix="/mcp")  # Refinement tool

# Mount MCP JSON-RPC server AFTER all routes are registered
# This way it doesn't override existing endpoints
from mcp_server import mcp_app
app.mount("/mcp/v1", mcp_app)