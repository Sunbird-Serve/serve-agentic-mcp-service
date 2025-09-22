from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from tools.wa_send import SendMessageInput, publish_wa_out
from tools import llm_extract, llm_reply, calendar_create

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
            {"name":"llm.extract_profile_fields"},
            {"name":"llm.generate_reply"},
            {"name":"calendar.create_event"}
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

app.include_router(llm_extract.router, prefix="/mcp")
app.include_router(llm_reply.router, prefix="/mcp")
app.include_router(calendar_create.router, prefix="/mcp")