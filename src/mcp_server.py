"""
MCP (Model Context Protocol) Server - JSON-RPC 2.0 Compliant

This module provides an MCP-compliant interface over the existing HTTP REST tools.
It implements the official MCP specification while keeping the HTTP REST API for backward compatibility.

Specification: https://modelcontextprotocol.io/specification
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
import httpx
import json
from datetime import datetime

# ------------- MCP PROTOCOL MODELS (JSON-RPC 2.0) -------------

class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[int | str] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[int | str] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 Error"""
    code: int
    message: str
    data: Optional[Any] = None

# MCP Standard Error Codes
class MCPErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

# ------------- MCP PROTOCOL TYPES -------------

class MCPServerCapabilities(BaseModel):
    """MCP Server Capabilities"""
    tools: Dict[str, Any] = {"listChanged": False}
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None

class MCPServerInfo(BaseModel):
    """MCP Server Information"""
    name: str
    version: str
    protocolVersion: str = "2024-11-05"
    capabilities: MCPServerCapabilities

class MCPTool(BaseModel):
    """MCP Tool Definition"""
    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any]

# ------------- MCP SERVER STATE -------------

class MCPServerState:
    """Maintains MCP server session state with timestamps and TTL tracking"""
    def __init__(self):
        self.initialized = False
        self.client_info: Optional[Dict] = None
        self.session_id: Optional[str] = None
        self.created_at = datetime.now()
        self.initialized_at: Optional[datetime] = None
        self.last_seen_at: Optional[datetime] = None
        self.request_count = 0
        self.tool_call_count = 0

    def initialize(self, client_info: Dict):
        """Initialize session with timestamp tracking"""
        self.initialized = True
        self.client_info = client_info
        self.session_id = f"session_{datetime.now().timestamp()}"
        self.initialized_at = datetime.now()
        self.last_seen_at = datetime.now()
    
    def update_activity(self):
        """Update last seen timestamp"""
        self.last_seen_at = datetime.now()
        self.request_count += 1
    
    def record_tool_call(self):
        """Record tool call for metrics"""
        self.tool_call_count += 1
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information with metrics"""
        if not self.initialized:
            return {"initialized": False}
        
        uptime = (datetime.now() - self.created_at).total_seconds()
        idle_time = (datetime.now() - self.last_seen_at).total_seconds() if self.last_seen_at else 0
        
        return {
            "session_id": self.session_id,
            "initialized": self.initialized,
            "client_info": self.client_info,
            "created_at": self.created_at.isoformat(),
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "uptime_seconds": uptime,
            "idle_seconds": idle_time,
            "request_count": self.request_count,
            "tool_call_count": self.tool_call_count
        }

# Global server state (per-instance)
mcp_state = MCPServerState()

# ------------- TOOL REGISTRY -------------

# Map MCP tool names to HTTP endpoints
TOOL_REGISTRY = {
    "wa.send_message": {
        "endpoint": "/mcp/wa.send_message",
        "method": "POST",
        "description": "Send WhatsApp message via Kafka",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Phone number (E.164 format)"},
                "text": {"type": "string", "description": "Message text"}
            },
            "required": ["to", "text"]
        }
    },
    "llm.call": {
        "endpoint": "/mcp/llm.call",
        "method": "POST",
        "description": "Generic LLM interface for any task",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt for the LLM"},
                "format": {"type": "string", "enum": ["json", "text"], "default": "json"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 1, "default": 0},
                "max_tokens": {"type": "integer", "default": 200},
                "context_window": {"type": "integer", "default": 1536},
                "threads": {"type": "integer", "default": 4}
            },
            "required": ["prompt"]
        }
    },
    "intent.detect": {
        "endpoint": "/mcp/intent.detect",
        "method": "POST",
        "description": "Detect user intent (confirmation/refinement/rejection)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "User's text"},
                "context": {"type": "string", "description": "Conversation context"}
            },
            "required": ["text"]
        }
    },
    "llm.extract_profile_fields": {
        "endpoint": "/mcp/llm.extract_profile_fields",
        "method": "POST",
        "description": "Extract teacher profile (subjects, grades, availability) - Hybrid approach",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "Conversation state"},
                "text": {"type": "string", "description": "User's text"},
                "locale": {"type": "string", "default": "en-IN"},
                "catalog": {"type": "object", "description": "Available subjects and grades"},
                "use_llm": {"type": "boolean", "default": True}
            },
            "required": ["text", "state"]
        }
    },
    "llm.generate_reply": {
        "endpoint": "/mcp/llm.generate_reply",
        "method": "POST",
        "description": "Generate conversational reply - Hybrid approach (template + LLM)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "enum": ["ask_next_question", "confirm", "fallback", "greeting", "help"]},
                "state": {"type": "string", "description": "Conversation state"},
                "profile_partial": {"type": "object", "default": {}},
                "locale": {"type": "string", "default": "en-IN"},
                "use_llm": {"type": "boolean", "default": True},
                "user_name": {"type": "string", "description": "User's name for personalization"}
            },
            "required": ["purpose", "state"]
        }
    },
    "llm.parse_time": {
        "endpoint": "/mcp/llm.parse_time",
        "method": "POST",
        "description": "Parse natural language time using LLM",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Time expression"},
                "tz": {"type": "string", "default": "Asia/Kolkata"},
                "duration_minutes": {"type": "integer", "default": 30},
                "now_iso": {"type": "string", "description": "Current time override (ISO format)"}
            },
            "required": ["text"]
        }
    },
    "time.parse_options": {
        "endpoint": "/mcp/time.parse_options",
        "method": "POST",
        "description": "Smart time parser - Fast rule-based + LLM fallback",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Time expression(s)"},
                "tz": {"type": "string", "default": "Asia/Kolkata"},
                "duration_minutes": {"type": "integer", "default": 30}
            },
            "required": ["text"]
        }
    },
    "time.refine_slots": {
        "endpoint": "/mcp/time.refine_slots",
        "method": "POST",
        "description": "Refine/modify time slots (replace, add, remove, shift)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "original_slots": {"type": "array", "description": "Previously returned slots"},
                "refinement_text": {"type": "string", "description": "Refinement request"},
                "tz": {"type": "string", "default": "Asia/Kolkata"},
                "duration_minutes": {"type": "integer", "default": 30}
            },
            "required": ["original_slots", "refinement_text"]
        }
    },
    "calendar.create_event": {
        "endpoint": "/mcp/calendar.create_event",
        "method": "POST",
        "description": "Create calendar event with meeting URL",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_iso": {"type": "string", "description": "Start time (ISO 8601)"},
                "end_iso": {"type": "string", "description": "End time (ISO 8601)"},
                "attendees": {"type": "array", "items": {"type": "string"}, "default": []},
                "timezone": {"type": "string", "default": "Asia/Kolkata"},
                "notes": {"type": "string"}
            },
            "required": ["title", "start_iso", "end_iso"]
        }
    }
}

# ------------- HTTP CLIENT FOR TOOL INVOCATION -------------

async def invoke_http_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke a tool by calling its HTTP endpoint.
    This bridges MCP protocol to our existing HTTP REST API.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool_info = TOOL_REGISTRY[tool_name]
    endpoint = tool_info["endpoint"]
    
    # Call the HTTP endpoint (use 127.0.0.1 to avoid routing issues)
    async with httpx.AsyncClient() as client:
        url = f"http://127.0.0.1:9000{endpoint}"
        print(f"[MCP Bridge] Calling {url} with args: {arguments}")
        response = await client.post(url, json=arguments, timeout=120.0)
        print(f"[MCP Bridge] Response status: {response.status_code}")
        response.raise_for_status()
        return response.json()

# ------------- MCP PROTOCOL HANDLERS -------------

async def handle_initialize(params: Optional[Dict]) -> Dict[str, Any]:
    """
    Handle initialize request.
    MCP spec: First message from client to establish connection.
    """
    if mcp_state.initialized:
        raise ValueError("Already initialized")
    
    client_info = params or {}
    mcp_state.initialize(client_info)
    
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": False},
            "logging": {}
        },
        "serverInfo": {
            "name": "serve-agentic-mcp-service",
            "version": "0.1.0"
        }
    }

async def handle_initialized(params: Optional[Dict]) -> None:
    """
    Handle initialized notification.
    MCP spec: Sent by client after receiving initialize response.
    """
    if not mcp_state.initialized:
        raise ValueError("Not initialized")
    # No response for notifications
    return None

async def handle_tools_list(params: Optional[Dict]) -> Dict[str, Any]:
    """
    Handle tools/list request.
    MCP spec: Return list of available tools.
    """
    if not mcp_state.initialized:
        raise ValueError("Not initialized - call initialize first")
    
    tools = []
    for tool_name, tool_info in TOOL_REGISTRY.items():
        tools.append({
            "name": tool_name,
            "description": tool_info["description"],
            "inputSchema": tool_info["inputSchema"]
        })
    
    return {"tools": tools}

async def handle_tools_call(params: Optional[Dict]) -> Dict[str, Any]:
    """
    Handle tools/call request.
    MCP spec: Invoke a specific tool with arguments.
    """
    if not mcp_state.initialized:
        raise ValueError("Not initialized - call initialize first")
    
    if not params:
        raise ValueError("Missing params")
    
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if not tool_name:
        raise ValueError("Missing tool name")
    
    # Record tool call for metrics
    mcp_state.record_tool_call()
    
    # Invoke the tool via HTTP
    try:
        result = await invoke_http_tool(tool_name, arguments)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2, ensure_ascii=False)
                }
            ]
        }
    except httpx.TimeoutException as e:
        raise ValueError(f"Tool timeout: {tool_name} took too long")
    except httpx.HTTPStatusError as e:
        raise ValueError(f"Tool error: {tool_name} returned {e.response.status_code}")
    except UnicodeEncodeError as e:
        raise ValueError(f"Unicode encoding error in tool response: {str(e)}")
    except Exception as e:
        raise ValueError(f"Tool execution failed: {str(e)}")

async def handle_ping(params: Optional[Dict]) -> Dict[str, Any]:
    """Handle ping request (keepalive)"""
    return {}

# ------------- METHOD ROUTER -------------

METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "initialized": handle_initialized,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
    "ping": handle_ping,
}

# ------------- MCP REQUEST PROCESSOR -------------

async def process_mcp_request(request: JSONRPCRequest) -> JSONRPCResponse:
    """
    Process a single MCP (JSON-RPC 2.0) request.
    """
    # Update activity timestamp
    mcp_state.update_activity()
    
    try:
        # Get handler for method
        handler = METHOD_HANDLERS.get(request.method)
        
        if not handler:
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(
                    code=MCPErrorCode.METHOD_NOT_FOUND,
                    message=f"Method not found: {request.method}",
                    data={"available_methods": list(METHOD_HANDLERS.keys())}
                ).dict()
            )
        
        # Call handler
        result = await handler(request.params)
        
        # Notifications (initialized) don't get responses
        if request.id is None:
            return None
        
        return JSONRPCResponse(
            id=request.id,
            result=result
        )
    
    except ValueError as e:
        return JSONRPCResponse(
            id=request.id,
            error=JSONRPCError(
                code=MCPErrorCode.INVALID_PARAMS,
                message=str(e),
                data={"param_error": str(e)}
            ).dict()
        )
    except httpx.TimeoutException as e:
        return JSONRPCResponse(
            id=request.id,
            error=JSONRPCError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message="Request timeout",
                data={"error": "Tool execution exceeded timeout limit"}
            ).dict()
        )
    except Exception as e:
        return JSONRPCResponse(
            id=request.id,
            error=JSONRPCError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message=f"Internal error: {str(e)}",
                data={"error_type": type(e).__name__}
            ).dict()
        )

# ------------- FASTAPI MCP ENDPOINT -------------

mcp_app = FastAPI(title="MCP Server", version="0.1.0")

@mcp_app.post("/jsonrpc")
async def mcp_endpoint(request: Request):
    """
    MCP JSON-RPC 2.0 endpoint with SSE support.
    
    Supports both JSON and Server-Sent Events (SSE) based on Accept header:
    - Accept: application/json → Standard JSON response
    - Accept: text/event-stream → SSE streaming response
    
    This is the main MCP-compliant endpoint that handles all protocol messages.
    Clients should send JSON-RPC 2.0 requests to this endpoint.
    """
    # Check Accept header for SSE
    accept_header = request.headers.get("accept", "application/json")
    use_sse = "text/event-stream" in accept_header
    
    try:
        body = await request.json()
        
        # SSE streaming (for long-running tools)
        if use_sse:
            async def event_stream():
                """Generate SSE events"""
                # Send initial event
                yield f"event: message\ndata: {json.dumps({'status': 'processing'})}\n\n"
                
                # Process request
                if isinstance(body, dict):
                    json_rpc_req = JSONRPCRequest(**body)
                    response = await process_mcp_request(json_rpc_req)
                    
                    if response:
                        yield f"event: message\ndata: {json.dumps(response.dict(exclude_none=True), ensure_ascii=False)}\n\n"
                
                # Send completion event
                yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
            
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Standard JSON response
        # Handle single request
        if isinstance(body, dict):
            json_rpc_req = JSONRPCRequest(**body)
            response = await process_mcp_request(json_rpc_req)
            
            # Notifications don't return responses
            if response is None:
                return JSONResponse(content=None, status_code=204)
            
            return JSONResponse(content=response.dict(exclude_none=True))
        
        # Handle batch requests (array of requests)
        elif isinstance(body, list):
            responses = []
            for req_data in body:
                json_rpc_req = JSONRPCRequest(**req_data)
                response = await process_mcp_request(json_rpc_req)
                if response:
                    responses.append(response.dict(exclude_none=True))
            
            return JSONResponse(content=responses)
        
        else:
            return JSONResponse(
                content=JSONRPCResponse(
                    error=JSONRPCError(
                        code=MCPErrorCode.INVALID_REQUEST,
                        message="Invalid request format"
                    ).dict()
                ).dict(),
                status_code=400
            )
    
    except Exception as e:
        return JSONResponse(
            content=JSONRPCResponse(
                error=JSONRPCError(
                    code=MCPErrorCode.PARSE_ERROR,
                    message=f"Parse error: {str(e)}"
                ).dict()
            ).dict(),
            status_code=400
        )

@mcp_app.get("/info")
async def mcp_info():
    """
    Get MCP server information with session metrics.
    Non-standard endpoint for debugging/discovery.
    """
    return {
        "name": "serve-agentic-mcp-service",
        "version": "0.1.0",
        "protocolVersion": "2024-11-05",
        "transport": "http",
        "supports": ["json", "sse"],
        "endpoints": {
            "jsonrpc": "/mcp/v1/jsonrpc",
            "info": "/mcp/v1/info",
            "health": "/mcp/v1/health"
        },
        "session": mcp_state.get_session_info(),
        "tools_count": len(TOOL_REGISTRY)
    }

@mcp_app.get("/health")
async def mcp_health():
    """Health check for MCP server"""
    return {
        "ok": True,
        "initialized": mcp_state.initialized,
        "uptime_seconds": (datetime.now() - mcp_state.created_at).total_seconds()
    }

