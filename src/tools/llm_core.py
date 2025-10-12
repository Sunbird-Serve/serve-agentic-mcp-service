"""
Core LLM Tool - Generic LLM interface for all MCP tools
"""
import json
import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, Optional

router = APIRouter()

# ------------- CONFIG -------------

OLLAMA_MODEL = "phi3:mini"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)

# ------------- MODEL DEFINITIONS -------------

class LLMRequest(BaseModel):
    prompt: str
    format: Optional[str] = "json"  # "json" or "text"
    temperature: float = 0
    max_tokens: int = 200
    context_window: int = 1536
    threads: int = 4

class LLMResponse(BaseModel):
    response: str
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

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
    
    Args:
        prompt: The prompt to send to the LLM
        format: "json" or "text" - controls LLM output format
        temperature: 0-1, controls randomness
        max_tokens: Maximum tokens to generate
        context_window: Context window size
        threads: Number of CPU threads
        model: Override default model
    
    Returns:
        tuple: (response_text, error_message)
        - On success: (response, None)
        - On failure: (None, error_message)
    """
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
        print(f"[llm.core] Calling {model or OLLAMA_MODEL} with format={format}, temp={temperature}")
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(OLLAMA_URL, json=body)
            resp.raise_for_status()
            payload = resp.json()
            response_text = (payload.get("response") or "").strip()
            
            if not response_text:
                return None, "LLM returned empty response"
            
            print(f"[llm.core] Success, received {len(response_text)} chars")
            return response_text, None
            
    except httpx.TimeoutException as e:
        error = f"LLM timeout after {HTTP_TIMEOUT.read}s"
        print(f"[llm.core] {error}")
        return None, error
    except Exception as e:
        error = f"LLM error: {type(e).__name__}: {str(e)}"
        print(f"[llm.core] {error}")
        return None, error

# ------------- MCP ENDPOINT -------------

@router.post("/llm.call", response_model=LLMResponse)
async def llm_call_endpoint(req: LLMRequest):
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

