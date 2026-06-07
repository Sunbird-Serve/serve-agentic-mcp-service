"""
MCP (Model Context Protocol) Server - JSON-RPC 2.0 Compliant

This module provides an MCP-compliant interface over the existing HTTP REST tools.
It implements the official MCP specification while keeping the HTTP REST API for backward compatibility.

Specification: https://modelcontextprotocol.io/specification
"""
from runtime_logging import configure_runtime_logging

# Configure UTF-8 console output and mirror all stdout/stderr logs to file.
configure_runtime_logging()

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
        "description": "Send WhatsApp message via Kafka. Supports template messages, interactive lists, buttons, or plain text. Priority: template > list > text/buttons",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Phone number (E.164 format)"},
                "template": {
                    "type": "object",
                    "description": "Optional WhatsApp template message. If provided, takes precedence over text/list/buttons",
                    "properties": {
                        "name": {"type": "string", "description": "Template name (e.g., 'serve_welcome')"},
                        "language": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "description": "Language code (e.g., 'en', 'hi')"}
                            },
                            "required": ["code"]
                        }
                    },
                    "required": ["name", "language"]
                },
                "text": {"type": "string", "description": "Message text (required if template/list is not provided)"},
                "buttons": {
                    "type": "array",
                    "description": "Optional list of buttons. Can be array of strings (quick replies) or array of button objects with id/title (used if template/list is not provided)",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique button ID"},
                                    "title": {"type": "string", "description": "Button label text"}
                                },
                                "required": ["id", "title"]
                            }
                        ]
                    }
                }
            },
            "required": ["to"]
        }
    },
    "llm.call": {
        "endpoint": "/mcp/llm.call",
        "method": "POST",
        "description": "General-purpose LLM call tool with messages array (system/user/assistant) for contextual responses",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                            "content": {"type": "string"}
                        },
                        "required": ["role", "content"]
                    },
                    "description": "Array of message objects with role and content"
                },
                "max_tokens": {"type": "integer", "default": 150, "description": "Maximum tokens to generate"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7, "description": "Sampling temperature"},
                "format": {"type": "string", "enum": ["json", "text"], "default": "text", "description": "Response format: json or text"}
            },
            "required": ["messages"]
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
    "llm.classify_response": {
        "endpoint": "/mcp/llm.classify_response",
        "method": "POST",
        "description": "Intelligently classify user responses to eligibility questions using LLM",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question_type": {"type": "string", "enum": ["consent", "age", "device", "commitment", "language_comfort"], "description": "Type of eligibility question"},
                "user_input": {"type": "string", "description": "User's response text"},
                "context": {"type": "object", "description": "Additional context (question_text, locale, etc.)"}
            },
            "required": ["question_type", "user_input"]
        }
    },
    "llm.handle_smart_edit": {
        "endpoint": "/mcp/llm.handle_smart_edit",
        "method": "POST",
        "description": "Intelligently handle teaching preference editing with context-aware understanding",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_history": {"type": "array", "items": {"type": "string"}, "description": "List of conversation messages"},
                "current_profile": {"type": "object", "description": "Current teaching profile to edit"},
                "user_input": {"type": "string", "description": "User's edit request text"}
            },
            "required": ["conversation_history", "current_profile", "user_input"]
        }
    },
    "llm.humanize_weekday_confirmation": {
        "endpoint": "/mcp/llm.humanize_weekday_confirmation",
        "method": "POST",
        "description": "Human-layer response for weekday 8–15 confirmation with strict JSON output",
        "inputSchema": {
            "type": "object",
            "properties": {
                "flow_state_summary": {"type": "string"},
                "user_input": {"type": "string"},
                "locale": {"type": "string", "default": "en-IN"}
            },
            "required": ["flow_state_summary", "user_input"]
        }
    },
    "faq.answer": {
        "endpoint": "/mcp/faq.answer",
        "method": "POST",
        "description": "Answer FAQs using KB retrieval and concise composition",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "policy_context": {"type": "string"},
                "kb_scope": {"type": "string", "default": "onboarding-basic"},
                "top_k": {"type": "integer", "default": 3},
                "state": {"type": "string", "description": "Optional: current onboarding state for routing"}
            },
            "required": ["question", "policy_context"]
        }
    },
    "knowledge.search": {
        "endpoint": "/mcp/knowledge.search",
        "method": "POST",
        "description": "Search knowledge base for FAQ snippets relevant to a user query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User's question or search query"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20, "description": "Number of snippets to return"},
                "policy_version": {"type": "string", "description": "Optional: policy version for filtering (e.g., 'v1.2')"}
            },
            "required": ["query"]
        }
    },
    "onboarding.handle_turn": {
        "endpoint": "/mcp/onboarding.handle_turn",
        "method": "POST",
        "description": "Parse a volunteer message, update onboarding facts, and return the next state with a ready reply",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "Current onboarding state (e.g., WELCOME, ELIGIBILITY_PART1)"},
                "message": {"type": "string", "description": "Volunteer WhatsApp message"},
                "locale": {"type": "string", "default": "en-IN"},
                "policy_version": {"type": "string"},
                "user_profile": {"type": "object"},
                "known_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": ["string", "number", "boolean", "object", "array", "null"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "source": {"type": "string"}
                        },
                        "required": ["type", "value"]
                    },
                    "description": "Previously collected facts for this session"
                }
            },
            "required": ["state", "message"]
        }
    },
    "llm.qa": {
        "endpoint": "/mcp/llm.qa",
        "method": "POST",
        "description": "Generate FAQ answer using LLM with RAG context from knowledge.search",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "User's question"},
                "snippets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["id", "title", "text"]
                    },
                    "description": "RAG context from knowledge.search"
                },
                "policy_version": {"type": "string", "description": "Optional policy version"},
                "knowledge_version": {"type": "string", "description": "Optional knowledge version"},
                "user_profile": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tz": {"type": "string"}
                    },
                    "description": "Optional user context"
                }
            },
            "required": ["question", "snippets"]
        }
    },
    "onboarding.parse_message": {
        "endpoint": "/mcp/onboarding.parse_message",
        "method": "POST",
        "description": "Parse onboarding messages for intents, consent, constraints, availability, eligibility hints",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "locale": {"type": "string", "default": "en-IN"},
                "state": {"type": "string", "description": "Optional: Current onboarding state for context-aware parsing"}
            },
            "required": ["text"]
        }
    },
    "consent.record": {
        "endpoint": "/mcp/consent.record",
        "method": "POST",
        "description": "Record volunteer consent for onboarding",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteerId": {"type": "string"},
                "consentGiven": {"type": "boolean"}
            },
            "required": ["volunteerId", "consentGiven"]
        }
    },
    "deferral.create": {
        "endpoint": "/mcp/deferral.create",
        "method": "POST",
        "description": "Create a deferral record when volunteer wants to postpone onboarding",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteerId": {"type": "string"},
                "reason": {"type": "string"},
                "until_ISO": {"type": "string", "description": "ISO8601 datetime when to contact again"},
                "idempotency_key": {"type": "string", "description": "Optional key to prevent duplicate deferrals"}
            },
            "required": ["volunteerId", "reason", "until_ISO"]
        }
    },
    "state.get": {
        "endpoint": "/mcp/state.get",
        "method": "POST",
        "description": "Get the current onboarding state for a volunteer (returns WELCOME if new)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteerId": {"type": "string"}
            },
            "required": ["volunteerId"]
        }
    },
    "state.advance": {
        "endpoint": "/mcp/state.advance",
        "method": "POST",
        "description": "Advance volunteer's onboarding state based on intent (with validation)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteerId": {"type": "string"},
                "intent": {"type": "string", "description": "Intent to advance (e.g., 'to_ELIGIBILITY_PART1')"},
                "idempotency_key": {"type": "string", "description": "Optional key to prevent duplicate transitions"}
            },
            "required": ["volunteerId", "intent"]
        }
    },
    "eligibility.check": {
        "endpoint": "/mcp/eligibility.check",
        "method": "POST",
        "description": "Check volunteer eligibility. For PART1 (age+device only), weeklyCommitmentHours can be omitted. For PART2, all fields required.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ageYears": {"type": "integer"},
                "hasDevice": {"type": "boolean"},
                "weeklyCommitmentHours": {"type": "number", "minimum": 0}
            },
            "required": ["ageYears", "hasDevice"]
        }
    },
    "preferences.save": {
        "endpoint": "/mcp/preferences.save",
        "method": "POST",
        "description": "Save volunteer preferences (days/time_windows/timezone)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteerId": {"type": "string"},
                "prefs": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "array", "items": {"type": "string"}},
                        "time_windows": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}, "required": ["start","end"]}
                                ]
                            }
                        },
                        "timezone": {"type": "string"}
                    },
                    "required": ["days", "time_windows", "timezone"]
                },
                "policy_version": {"type": "string"},
                "idempotency_key": {"type": "string"}
            },
            "required": ["volunteerId", "prefs"]
        }
    },
    "policy.scheduling": {
        "endpoint": "/mcp/policy.scheduling",
        "method": "POST",
        "description": "Return scheduling policy (weekend gate, blackout dates, version)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "region_id": {"type": "string"}
            }
        }
    },
    "prefs.confirmation_polish": {
        "endpoint": "/mcp/prefs.confirmation_polish",
        "method": "POST",
        "description": "Generate one-line WhatsApp confirmation for parsed day/time prefs",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "array", "items": {"type": "string"}},
                "time_windows": {"type": "array", "items": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}, "required": ["start","end"]}},
                "timezone": {"type": "string"},
                "weekend_gate": {"type": "boolean"}
            },
            "required": ["days","time_windows","timezone","weekend_gate"]
        }
    },
    "intent.classify_orientation": {
        "endpoint": "/mcp/intent.classify_orientation",
        "method": "POST",
        "description": "Classify if text refers to orientation or class (or unclear)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "state": {"type": "string"}
            },
            "required": ["text"]
        }
    },
    "slots.propose": {
        "endpoint": "/mcp/slots.propose",
        "method": "POST",
        "description": "Propose slots based on time band and day preferences. If daysWhitelist is null, no day filter is applied (server defaults to weekdays). If seedTimeIso is provided, the server infers an appropriate band and centers proposals around that time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteerId": {"type": "string"},
                "timeBand": {"type": ["string","null"], "description": "Optional. '8-11' | '12-15' | 'MORNING' | 'AFTERNOON'"},
                "daysWhitelist": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"}
                    ],
                    "description": "Optional list of weekdays (Mon, Tue, etc.). If null, returns slots across all weekdays."
                },
                "limit": {"type": "integer", "default": 2},
                "seedTimeIso": {"type": ["string", "null"], "description": "Optional ISO time to seed proposals (e.g., parsed 'Tomorrow 3 pm')"},
                "seedTimesIso": {"type": "array", "items": {"type": "string"}, "description": "Optional list of ISO seed times to propose directly"},
                "tz": {"type": ["string", "null"], "default": "Asia/Kolkata", "description": "Timezone for seedTimeIso"}
            },
            "required": ["volunteerId"]
        }
    },
    "slot.hold": {
        "endpoint": "/mcp/slot.hold",
        "method": "POST",
        "description": "Temporarily hold a slot",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slotId": {"type": "string"}
            },
            "required": ["slotId"]
        }
    },
    "slot.book": {
        "endpoint": "/mcp/slot.book",
        "method": "POST",
        "description": "Confirm a held slot",
        "inputSchema": {
            "type": "object",
            "properties": {
                "holdId": {"type": "string"}
            },
            "required": ["holdId"]
        }
    },
    "reminder.create": {
        "endpoint": "/mcp/reminder.create",
        "method": "POST",
        "description": "Create a reminder",
        "inputSchema": {
            "type": "object",
            "properties": {
                "when_ISO": {"type": "string"},
                "reason": {"type": "string"},
                "volunteerId": {"type": "string"}
            },
            "required": ["when_ISO", "reason"]
        }
    },
    "telemetry.emit": {
        "endpoint": "/mcp/telemetry.emit",
        "method": "POST",
        "description": "Emit telemetry event",
        "inputSchema": {
            "type": "object",
            "properties": {
                "event": {"type": "string"},
                "payload": {"type": "object"}
            },
            "required": ["event", "payload"]
        }
    },
    "onboarding.next": {
        "endpoint": "/mcp/onboarding.next",
        "method": "POST",
        "description": "[DEPRECATED] Use discrete tools instead: onboarding.parse_message, eligibility.check, llm.call, etc. Agent should own conversation orchestration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "object"},
                "user_text": {"type": "string"},
                "locale": {"type": "string", "default": "en-IN"}
            },
            "required": ["session", "user_text"]
        },
        "deprecated": True
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
    "time.parser_refine": {
        "endpoint": "/mcp/time.parser_refine",
        "method": "POST",
        "description": "Parse vague day/time phrases into weekday-constrained slots",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "policy": {
                    "type": "object",
                    "properties": {
                        "weekday_only": {"type": "boolean"},
                        "window_24h": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"}
                            },
                            "required": ["start", "end"]
                        },
                        "map_phrases": {"type": "boolean"}
                    },
                    "required": ["weekday_only", "window_24h", "map_phrases"]
                },
                "need_at_least": {"type": "integer", "default": 2},
                "locale": {"type": "string", "default": "en-IN"}
            },
            "required": ["text", "policy"]
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
    },
    "serve.needs.list": {
        "endpoint": "/mcp/serve.needs.list",
        "method": "POST",
        "description": "Fetch approved Serve needs for volunteer discovery",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "default": 0, "minimum": 0, "description": "Page number (0-indexed)"},
                "size": {"type": "integer", "default": 10, "minimum": 1, "maximum": 20, "description": "Page size (max 20)"},
                "status": {"type": "string", "default": "Approved", "description": "Filter by status"}
            }
        }
    },
    "serve.fulfill.nominate": {
        "endpoint": "/mcp/serve.fulfill.nominate",
        "method": "POST",
        "description": "Nominate a volunteer for a Serve need",
        "inputSchema": {
            "type": "object",
            "properties": {
                "needId": {"type": "string", "description": "UUID string of the Serve need"},
                "nominatedUserId": {"type": "string", "description": "UUID string of the nominated volunteer/user"},
                "source": {"type": "string", "description": "Source of nomination: 'whatsapp', 'portal', 'other'"},
                "idempotency_key": {"type": "string", "description": "Optional idempotency key to prevent duplicate nominations"}
            },
            "required": ["needId", "nominatedUserId"]
        }
    },
    "serve.whatsapp.send_class_video": {
        "endpoint": "/mcp/serve.whatsapp.send_class_video",
        "method": "POST",
        "description": "Send an in-app playable WhatsApp class video (MP4). The video file is hosted on the MCP server. Uploads video to WhatsApp Media API and sends using media_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to_phone": {"type": "string", "description": "Recipient phone number (E.164 format)"},
                "caption": {"type": "string", "description": "Optional caption text for the video"}
            },
            "required": ["to_phone"]
        }
    },
    "serve.whatsapp.send_welcome_video": {
        "endpoint": "/mcp/serve.whatsapp.send_welcome_video",
        "method": "POST",
        "description": "Send an in-app playable WhatsApp welcome video (MP4). The video file is hosted on the MCP server. Uploads video to WhatsApp Media API and sends using media_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to_phone": {"type": "string", "description": "Recipient phone number (E.164 format)"},
                "caption": {"type": "string", "description": "Optional caption text for the video"}
            },
            "required": ["to_phone"]
        }
    },
    "serve.whatsapp.send_thankyou_video": {
        "endpoint": "/mcp/serve.whatsapp.send_thankyou_video",
        "method": "POST",
        "description": "Send an in-app playable WhatsApp thank you video (MP4). The video file is hosted on the MCP server. Uploads video to WhatsApp Media API and sends using media_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to_phone": {"type": "string", "description": "Recipient phone number (E.164 format)"},
                "caption": {"type": "string", "description": "Optional caption text for the video"}
            },
            "required": ["to_phone"]
        }
    },
    "serve.volunteer.email_exists": {
        "endpoint": "/mcp/serve.volunteer.email_exists",
        "method": "POST",
        "description": "Check if a volunteer email exists in SERVE system and get volunteer ID if found",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email address to check"}
            },
            "required": ["email"]
        }
    },
    "serve.volunteer.register": {
        "endpoint": "/mcp/serve.volunteer.register",
        "method": "POST",
        "description": "Register a new volunteer in SERVE system. Calls two APIs: Create User, then Create User Profile. Should only be called when email_exists returned exists=false",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name of the volunteer"},
                "email": {"type": "string", "description": "Email address"},
                "wa_phone": {"type": "string", "description": "WhatsApp phone number with country code"},
                "day_preferred": {"type": "array", "items": {"type": "string"}, "description": "Preferred days, e.g. ['Monday', 'Wednesday']"},
                "time_preferred": {"type": "array", "items": {"type": "string"}, "description": "Preferred times, e.g. ['Morning', 'Afternoon']"},
                "agency_id": {"type": "string", "description": "Agency ID (uses default if not provided)"},
                "idempotency_key": {"type": "string", "description": "Idempotency key for safe retries"}
            },
            "required": ["name", "email", "wa_phone", "idempotency_key"]
        }
    },
    "firebase.auth.email_exists": {
        "endpoint": "/mcp/firebase.auth.email_exists",
        "method": "POST",
        "description": "Check if a Firebase user exists by email address",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email address to check"}
            },
            "required": ["email"]
        }
    },
    "firebase.auth.ensure_user": {
        "endpoint": "/mcp/firebase.auth.ensure_user",
        "method": "POST",
        "description": "Idempotently ensure a Firebase email/password user exists. Creates user if missing and optionally sends password reset email via Firebase REST API",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email address"},
                "display_name": {"type": "string", "description": "Display name for the user"},
                "create_if_missing": {"type": "boolean", "default": True, "description": "Create user if doesn't exist"},
                "generate_reset_link": {"type": "boolean", "default": True, "description": "Send password reset email"}
            },
            "required": ["email"]
        }
    },
    "serve.volunteer.update_status": {
        "endpoint": "/mcp/serve.volunteer.update_status",
        "method": "POST",
        "description": "Update SERVE volunteer lifecycle status after selection decision. Idempotent and safe to retry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteer_id": {"type": "string", "description": "SERVE volunteer osid"},
                "status": {"type": "string", "enum": ["RECOMMENDED", "ONHOLD"], "description": "New status: RECOMMENDED or ONHOLD"},
                "send": {"type": "boolean", "default": True, "description": "Send notification (default: true)"}
            },
            "required": ["volunteer_id", "status"]
        }
    },
    "fulfill.match": {
        "endpoint": "/mcp/fulfill.match",
        "method": "POST",
        "description": "Match a volunteer to best-fit Serve needs using rule-based scoring (day overlap 45%, time overlap 35%, recency 20%). Returns ranked matches with score breakdown.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volunteerId": {"type": "string", "description": "Volunteer ID"},
                "preferences": {
                    "type": "object",
                    "description": "Volunteer preferences (optional — fetched from store if not provided)",
                    "properties": {
                        "days": {"type": "array", "items": {"type": "string"}, "description": "Preferred days, e.g. ['Mon','Tue']"},
                        "time_windows": {"type": "array", "items": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}}, "description": "Preferred time windows"},
                        "timezone": {"type": "string", "default": "Asia/Kolkata"}
                    },
                    "required": ["days"]
                },
                "maxResults": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20, "description": "Maximum matches to return"},
                "minScore": {"type": "number", "default": 0.1, "minimum": 0, "maximum": 1, "description": "Minimum score threshold"}
            },
            "required": ["volunteerId"]
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
        
        # Explicitly decode as UTF-8 to handle Unicode characters (emojis, etc.)
        # This ensures Windows 'charmap' encoding doesn't interfere
        response.encoding = "utf-8"
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
        
        # Safely serialize result to JSON with proper Unicode handling
        # Ensure all strings in the result are properly encoded as UTF-8
        def ensure_utf8_str(obj):
            """Recursively ensure all strings are UTF-8 compatible"""
            if isinstance(obj, dict):
                return {k: ensure_utf8_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_utf8_str(item) for item in obj]
            elif isinstance(obj, str):
                # Ensure string can be encoded as UTF-8 (should always be true for Python 3 strings)
                # This is mainly for validation - Python 3 strings are already Unicode
                try:
                    obj.encode('utf-8')
                    return obj
                except UnicodeEncodeError:
                    # Fallback: replace problematic characters
                    return obj.encode('utf-8', errors='replace').decode('utf-8')
            else:
                return obj
        
        # Normalize result to ensure UTF-8 compatibility
        safe_result = ensure_utf8_str(result)
        
        # Serialize with explicit UTF-8 handling
        # ensure_ascii=False allows Unicode characters to be preserved
        json_text = json.dumps(safe_result, indent=2, ensure_ascii=False)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json_text
                }
            ]
        }
    except httpx.TimeoutException as e:
        raise ValueError(f"Tool timeout: {tool_name} took too long")
    except httpx.HTTPStatusError as e:
        raise ValueError(f"Tool error: {tool_name} returned {e.response.status_code}")
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        # Safely encode error message to avoid encoding errors when creating the ValueError
        try:
            error_msg = str(e)
        except UnicodeEncodeError:
            # If str(e) fails, use a safe ASCII representation
            error_msg = f"Unicode encoding error: {type(e).__name__}"
        raise ValueError(f"Unicode encoding error in tool response: {error_msg}")
    except Exception as e:
        # Safely encode error message to avoid encoding errors
        try:
            error_msg = str(e)
        except UnicodeEncodeError:
            error_msg = f"Error: {type(e).__name__}"
        raise ValueError(f"Tool execution failed: {error_msg}")

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
        # Safely encode error message to avoid encoding errors
        try:
            error_msg = str(e)
        except (UnicodeEncodeError, UnicodeDecodeError):
            error_msg = f"Invalid parameters: {type(e).__name__}"
        except Exception:
            error_msg = "Invalid parameters"
        
        return JSONRPCResponse(
            id=request.id,
            error=JSONRPCError(
                code=MCPErrorCode.INVALID_PARAMS,
                message=error_msg,
                data={"param_error": error_msg}
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
        # Safely encode error message to avoid encoding errors
        try:
            error_msg = str(e)
        except (UnicodeEncodeError, UnicodeDecodeError):
            error_msg = f"Internal error: {type(e).__name__}"
        except Exception:
            error_msg = "Internal error: Unknown exception"
        
        return JSONRPCResponse(
            id=request.id,
            error=JSONRPCError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message=error_msg,
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
                yield f"event: message\ndata: {json.dumps({'status': 'processing'}, ensure_ascii=False)}\n\n"
                
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
            
            # Convert response to dict - Pydantic already handles Unicode properly
            response_dict = response.dict(exclude_none=True)
            
            # FastAPI JSONResponse handles UTF-8 correctly when charset is specified
            return JSONResponse(
                content=response_dict, 
                media_type="application/json; charset=utf-8"
            )
        
        # Handle batch requests (array of requests)
        elif isinstance(body, list):
            responses = []
            for req_data in body:
                json_rpc_req = JSONRPCRequest(**req_data)
                response = await process_mcp_request(json_rpc_req)
                if response:
                    responses.append(response.dict(exclude_none=True))
            
            return JSONResponse(
                content=responses, 
                media_type="application/json; charset=utf-8"
            )
        
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
        # Safely encode error message to avoid encoding errors
        try:
            error_msg = str(e)
        except (UnicodeEncodeError, UnicodeDecodeError):
            error_msg = f"Parse error: {type(e).__name__}"
        except Exception:
            error_msg = "Parse error: Unknown exception"
        
        return JSONResponse(
            content=JSONRPCResponse(
                error=JSONRPCError(
                    code=MCPErrorCode.PARSE_ERROR,
                    message=error_msg
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

