"""
Core LLM Tool - Generic LLM interface for all MCP tools

Supports multiple LLM providers:
- Anthropic Claude (API)
- Ollama (local)
"""
import json
import httpx
import os
from fastapi import APIRouter
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, Optional, Literal, List

router = APIRouter()

# ------------- CONFIG -------------

class LLMSettings(BaseSettings):
    """LLM Configuration from environment"""
    LLM_PROVIDER: str = "claude"
    CLAUDE_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    OLLAMA_MODEL: str = "phi3:mini"
    OLLAMA_URL: str = "http://127.0.0.1:11434/api/generate"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Load settings
llm_settings = LLMSettings()

# Extract settings
LLM_PROVIDER = llm_settings.LLM_PROVIDER
CLAUDE_API_KEY = llm_settings.CLAUDE_API_KEY
CLAUDE_MODEL = llm_settings.CLAUDE_MODEL
OLLAMA_MODEL = llm_settings.OLLAMA_MODEL
OLLAMA_URL = llm_settings.OLLAMA_URL
CLAUDE_URL = "https://api.anthropic.com/v1/messages"

# Debug: Print configuration on module load
print(f"[llm.core] CONFIG: Provider={LLM_PROVIDER}, Claude API Key={'SET (' + CLAUDE_API_KEY[:20] + '...)' if CLAUDE_API_KEY else 'NOT SET'}, Model={CLAUDE_MODEL}")

# Timeouts
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)

# ------------- MODEL DEFINITIONS -------------

class Message(BaseModel):
    """Message object for conversation context"""
    role: Literal["system", "user", "assistant"]
    content: str

class LLMRequest(BaseModel):
    prompt: str
    format: Optional[str] = "json"  # "json" or "text"
    temperature: float = 0
    max_tokens: int = 200
    context_window: int = 1536
    threads: int = 4

class LLMMessagesRequest(BaseModel):
    """New messages-based request format (per spec)"""
    messages: List[Message]
    max_tokens: int = 150
    temperature: float = 0.7
    format: Optional[str] = "text"  # "json" or "text" (default: "text")

class LLMResponse(BaseModel):
    response: str
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

class LLMContentResponse(BaseModel):
    """Response format with content field (per spec)"""
    content: Optional[str] = None
    message: Optional[str] = None  # Alternative field name
    text: Optional[str] = None  # Alternative field name
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # Token usage: {prompt_tokens, completion_tokens, total_tokens}
    model: Optional[str] = None  # Model used (e.g., "claude-3-5-haiku-latest")

# ------------- JSON EXTRACTOR -------------

import re
_JSON_BLOCK = re.compile(r'(\{.*\}|\[.*\])', re.S)

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from LLM response that might have markdown or extra text.
    """
    if not text:
        return None
    t = text.strip().strip('`').strip()
    
    # Try to parse as-is first
    try:
        json.loads(t)
        return t
    except:
        pass
    
    # Extract JSON block
    m = _JSON_BLOCK.search(t)
    if not m:
        return None
    
    block = m.group(1)
    
    # Auto-fix unbalanced braces
    open_braces = block.count("{")
    close_braces = block.count("}")
    if open_braces != close_braces:
        if open_braces > close_braces:
            block += "}" * (open_braces - close_braces)
        elif close_braces > open_braces:
            block = block[:-1 * (close_braces - open_braces)]
    
    return block

# ------------- CORE LLM FUNCTION -------------

async def call_llm(
    prompt: str,
    format: str = "json",
    temperature: float = 0,
    max_tokens: int = 200,
    context_window: int = 1536,
    threads: int = 4,
    model: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Generic LLM caller that can be used by any tool.
    Supports Claude API and Ollama.
    
    Args:
        prompt: The prompt to send to the LLM
        format: "json" or "text" - controls LLM output format
        temperature: 0-1, controls randomness
        max_tokens: Maximum tokens to generate
        context_window: Context window size (Ollama only)
        threads: Number of CPU threads (Ollama only)
        model: Override default model
    
    Returns:
        tuple: (response_text, error_message)
        - On success: (response, None)
        - On failure: (None, error_message)
    """
    provider = LLM_PROVIDER.lower()
    
    # Route to appropriate provider
    if provider == "claude":
        return await _call_claude(prompt, format, temperature, max_tokens, model)
    elif provider == "ollama":
        return await _call_ollama(prompt, format, temperature, max_tokens, context_window, threads, model)
    else:
        return None, f"Unknown LLM provider: {provider}"

# ------------- CLAUDE API -------------

async def _call_claude_with_messages(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    model: Optional[str] = None,
    format: str = "text"
) -> tuple[Optional[str], Optional[str], Optional[Dict[str, int]], Optional[str]]:
    """
    Call Claude API with messages array format.
    
    Returns: (response_text, error, usage, model_name)
    """
    if not CLAUDE_API_KEY:
        return None, "CLAUDE_API_KEY not set in environment", None, None
    
    # Separate system messages from user/assistant messages
    system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
    system_text = "\n".join(system_messages) if system_messages else None
    
    # Filter out system messages for Claude (they go in system field)
    claude_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg["role"] in ["user", "assistant"]
    ]
    
    body = {
        "model": model or CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": claude_messages
    }
    
    if system_text:
        body["system"] = system_text
    
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    try:
        print(f"[llm.core] Calling Claude {model or CLAUDE_MODEL} with {len(messages)} messages (temp={temperature})")
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(CLAUDE_URL, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            
            # Extract text from Claude response
            content_blocks = payload.get("content", [])
            if not content_blocks:
                return None, "Claude returned empty response", None, None
            
            response_text = content_blocks[0].get("text", "").strip()
            
            if not response_text:
                return None, "Claude returned empty text", None, None
            
            # Extract usage information if available
            usage_info = payload.get("usage")
            usage = None
            if usage_info:
                usage = {
                    "prompt_tokens": usage_info.get("input_tokens", 0),
                    "completion_tokens": usage_info.get("output_tokens", 0),
                    "total_tokens": usage_info.get("input_tokens", 0) + usage_info.get("output_tokens", 0)
                }
            
            model_name = payload.get("model") or model or CLAUDE_MODEL
            
            # Validate JSON if format="json"
            if format == "json":
                try:
                    import json
                    json.loads(response_text)  # Validate JSON
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    json_text = extract_json_from_text(response_text)
                    if json_text:
                        response_text = json_text
                    else:
                        return None, "Claude returned invalid JSON format", usage, model_name
            
            print(f"[llm.core] Claude success, received {len(response_text)} chars")
            return response_text, None, usage, model_name
            
    except httpx.TimeoutException as e:
        error = f"Claude timeout after {HTTP_TIMEOUT.read}s"
        print(f"[llm.core] {error}")
        return None, error, None, None
    except httpx.HTTPStatusError as e:
        # Retry with -latest alias if model not found
        status = e.response.status_code
        text = e.response.text
        if status == 404:
            base = (model or CLAUDE_MODEL)
            latest = None
            if base and '-' in base:
                # Convert e.g., claude-3-5-sonnet-20241022 -> claude-3-5-sonnet-latest
                parts = base.split('-')
                if parts[-1].isdigit():
                    latest = '-'.join(parts[:-1] + ['latest'])
            if latest:
                print(f"[llm.core] Model not found, retrying with alias: {latest}")
                body["model"] = latest
                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                        resp = await client.post(CLAUDE_URL, json=body, headers=headers)
                        resp.raise_for_status()
                        payload = resp.json()
                        content_blocks = payload.get("content", [])
                        if not content_blocks:
                            return None, "Claude returned empty response", None, None
                        response_text = content_blocks[0].get("text", "").strip()
                        if not response_text:
                            return None, "Claude returned empty text", None, None
                        usage_info = payload.get("usage")
                        usage = None
                        if usage_info:
                            usage = {
                                "prompt_tokens": usage_info.get("input_tokens", 0),
                                "completion_tokens": usage_info.get("output_tokens", 0),
                                "total_tokens": usage_info.get("input_tokens", 0) + usage_info.get("output_tokens", 0)
                            }
                        model_name = payload.get("model") or latest
                        print(f"[llm.core] Claude success (alias), received {len(response_text)} chars")
                        return response_text, None, usage, model_name
                except Exception as e2:
                    err2 = f"Claude alias retry failed: {type(e2).__name__}: {str(e2)}"
                    print(f"[llm.core] {err2}")
                    # Fall through to original error
        error = f"Claude API error {status}: {text}"
        print(f"[llm.core] {error}")
        return None, error, None, None
    except Exception as e:
        error = f"Claude error: {type(e).__name__}: {str(e)}"
        print(f"[llm.core] {error}")
        return None, error, None, None

async def _call_claude(
    prompt: str,
    format: str,
    temperature: float,
    max_tokens: int,
    model: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Call Anthropic Claude API"""
    
    if not CLAUDE_API_KEY:
        return None, "CLAUDE_API_KEY not set in environment"
    
    # Note: JSON format requirement should be handled in the caller's system message
    # or in the endpoint handler before calling this function
    
    body = {
        "model": model or CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    # For prompt-based calls, we do not inject a separate system message here.
    # If a system instruction is required, the caller should include it in the prompt
    # or use the messages-based llm.call endpoint which supports a system role.
    
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    try:
        print(f"[llm.core] Calling Claude {model or CLAUDE_MODEL} (format={format}, temp={temperature})")
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(CLAUDE_URL, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            
            # Extract text from Claude response
            content_blocks = payload.get("content", [])
            if not content_blocks:
                return None, "Claude returned empty response"
            
            response_text = content_blocks[0].get("text", "").strip()
            
            if not response_text:
                return None, "Claude returned empty text"
            
            print(f"[llm.core] Claude success, received {len(response_text)} chars")
            return response_text, None
            
    except httpx.TimeoutException as e:
        error = f"Claude timeout after {HTTP_TIMEOUT.read}s"
        print(f"[llm.core] {error}")
        return None, error
    except httpx.HTTPStatusError as e:
        # Retry with -latest alias if model not found
        status = e.response.status_code
        text = e.response.text
        if status == 404:
            base = (model or CLAUDE_MODEL)
            latest = None
            if base and '-' in base:
                parts = base.split('-')
                if parts[-1].isdigit():
                    latest = '-'.join(parts[:-1] + ['latest'])
            if latest:
                print(f"[llm.core] Model not found, retrying with alias: {latest}")
                body = {
                    "model": latest,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                if format == "json":
                    body["system"] = "You must respond with valid JSON only. No markdown, no explanations, just the JSON object or array."
                headers = {
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                        resp = await client.post(CLAUDE_URL, json=body, headers=headers)
                        resp.raise_for_status()
                        payload = resp.json()
                        content_blocks = payload.get("content", [])
                        if not content_blocks:
                            return None, "Claude returned empty response"
                        response_text = content_blocks[0].get("text", "").strip()
                        if not response_text:
                            return None, "Claude returned empty text"
                        print(f"[llm.core] Claude success (alias), received {len(response_text)} chars")
                        return response_text, None
                except Exception as e2:
                    err2 = f"Claude alias retry failed: {type(e2).__name__}: {str(e2)}"
                    print(f"[llm.core] {err2}")
        error = f"Claude API error {status}: {text}"
        print(f"[llm.core] {error}")
        return None, error
    except Exception as e:
        error = f"Claude error: {type(e).__name__}: {str(e)}"
        print(f"[llm.core] {error}")
        return None, error

# ------------- OLLAMA API -------------

async def _call_ollama(
    prompt: str,
    format: str,
    temperature: float,
    max_tokens: int,
    context_window: int,
    threads: int,
    model: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Call local Ollama API"""
    
    body = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "format": format if format == "json" else None,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "num_ctx": context_window,
            "num_thread": threads,
            "keep_alive": "2m",
        },
        "stream": False
    }

    try:
        print(f"[llm.core] Calling Ollama {model or OLLAMA_MODEL} (format={format}, temp={temperature})")
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(OLLAMA_URL, json=body)
            resp.raise_for_status()
            payload = resp.json()
            response_text = (payload.get("response") or "").strip()
            
            if not response_text:
                return None, "Ollama returned empty response"
            
            print(f"[llm.core] Ollama success, received {len(response_text)} chars")
            return response_text, None
            
    except httpx.TimeoutException as e:
        error = f"Ollama timeout after {HTTP_TIMEOUT.read}s"
        print(f"[llm.core] {error}")
        return None, error
    except Exception as e:
        error = f"Ollama error: {type(e).__name__}: {str(e)}"
        print(f"[llm.core] {error}")
        return None, error

# ------------- MCP ENDPOINTS -------------

@router.post("/llm.call", response_model=LLMContentResponse)
async def llm_call_endpoint(req: LLMMessagesRequest):
    """
    MCP Tool: llm.call (messages-based format)
    
    General-purpose LLM call tool that accepts a messages array (system/user/assistant)
    and generates a text response.
    
    Used for contextual persuasion, personalized responses, and other conversational tasks.
    
    Args:
        messages: Array of message objects with role (system/user/assistant) and content
        max_tokens: Maximum tokens to generate (default: 150)
        temperature: Sampling temperature 0.0-1.0 (default: 0.7)
    
    Returns:
        Response with "content" field containing generated text
    """
    try:
        # Convert Pydantic models to dict format for Claude API
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in req.messages]
        
        # Get format parameter (default: "text")
        format_type = req.format or "text"
        
        # If format="json", add JSON requirement to system message if not already present
        if format_type == "json":
            has_system = any(msg["role"] == "system" for msg in messages_dict)
            has_json_hint = any("JSON" in msg["content"].upper() for msg in messages_dict if msg["role"] == "system")
            if not has_json_hint:
                if has_system:
                    # Add JSON requirement to existing system message
                    for msg in messages_dict:
                        if msg["role"] == "system":
                            msg["content"] = f"{msg['content']}\n\nYou must respond with valid JSON only. No markdown, no explanations, just the JSON object or array."
                            break
                else:
                    # Add new system message with JSON requirement
                    messages_dict.insert(0, {
                        "role": "system",
                        "content": "You must respond with valid JSON only. No markdown, no explanations, just the JSON object or array."
                    })
        
        # Call Claude with messages
        response_text, error, usage, model_name = await _call_claude_with_messages(
            messages=messages_dict,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            format=format_type
        )
        
        if error:
            return LLMContentResponse(
                content=None,
                error=error,
                usage=usage,
                model=model_name
            )
        
        # Return in content field (per spec)
        return LLMContentResponse(
            content=response_text,
            message=response_text,  # Alternative field for backward compatibility
            text=response_text,  # Alternative field for backward compatibility
            usage=usage,
            model=model_name
        )
        
    except Exception as e:
        return LLMContentResponse(
            content=None,
            error=f"Error processing request: {str(e)}"
        )

@router.post("/llm.call.prompt", response_model=LLMResponse)
async def llm_call_prompt_endpoint(req: LLMRequest):
    """
    MCP Tool: llm.call
    
    Generic LLM interface that can be used by any tool or agent.
    
    Examples:
    1. JSON extraction:
       {
         "prompt": "Extract name and age from: 'Hi I'm John, 25 years old'. Return JSON.",
         "format": "json",
         "temperature": 0
       }
    
    2. Text generation:
       {
         "prompt": "Write a friendly greeting message",
         "format": "text",
         "temperature": 0.7
       }
    
    3. Structured parsing:
       {
         "prompt": "Parse 'tomorrow evening' into ISO datetime. Current time: 2025-10-09T14:00:00+05:30",
         "format": "json",
         "temperature": 0
       }
    """
    response_text, error = await call_llm(
        prompt=req.prompt,
        format=req.format,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        context_window=req.context_window,
        threads=req.threads
    )
    
    if error:
        return LLMResponse(
            response="",
            success=False,
            error=error
        )
    
    return LLMResponse(
        response=response_text,
        success=True,
        raw_response={"length": len(response_text)}
    )

# ------------- HELPER FUNCTIONS FOR TOOLS -------------

async def call_llm_for_json(prompt: str, **kwargs) -> tuple[Optional[Dict], Optional[str]]:
    """
    Helper: Call LLM and parse JSON response.
    Returns: (parsed_dict, error_message)
    """
    response_text, error = await call_llm(prompt, format="json", **kwargs)
    
    if error:
        return None, error
    
    # Try to extract and parse JSON
    json_text = extract_json_from_text(response_text) or response_text
    
    try:
        data = json.loads(json_text)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {str(e)}"

async def call_llm_for_text(prompt: str, **kwargs) -> tuple[Optional[str], Optional[str]]:
    """
    Helper: Call LLM for text generation.
    Returns: (text, error_message)
    """
    return await call_llm(prompt, format="text", **kwargs)

async def call_llm_for_json_messages(
    system_text: str,
    user_text: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 220,
    assistant_history: Optional[List[Dict[str, str]]] = None
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Helper: Call LLM with messages array (system + user[/assistant]) and parse JSON response.
    Returns: (parsed_dict, error_message)
    """
    messages: List[Dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if assistant_history:
        for msg in assistant_history:
            if msg.get("role") in ("assistant",):
                messages.append({"role": "assistant", "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_text})

    response_text, error = await _call_claude_with_messages(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    if error:
        return None, error

    json_text = extract_json_from_text(response_text) or response_text
    try:
        data = json.loads(json_text)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {str(e)}"

