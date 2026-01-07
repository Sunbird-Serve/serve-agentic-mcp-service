"""
Serve Fulfillment Tool
- Nominate volunteers for Serve needs
- Handle nomination API calls with proper error handling
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import httpx

from config import settings

router = APIRouter()

# --------- Constants ---------

# Get base URL from settings (reads from env var SERVE_BASE_URL with default)
SERVE_BASE_URL = settings.SERVE_BASE_URL
HTTP_TIMEOUT = 10.0  # seconds (connect + read)

# --------- Models ---------

class NominateRequest(BaseModel):
    """Request to nominate a volunteer for a need"""
    needId: str = Field(..., description="UUID string of the Serve need")
    nominatedUserId: str = Field(..., description="UUID string of the nominated volunteer/user")
    source: Optional[str] = Field(
        None,
        description="Source of nomination: 'whatsapp', 'portal', 'other'"
    )
    idempotency_key: Optional[str] = Field(
        None,
        description="Optional idempotency key to prevent duplicate nominations"
    )

class NominateResponse(BaseModel):
    """Normalized response from nomination API"""
    ok: bool
    needId: str
    nominatedUserId: str
    status_code: int
    raw: Dict[str, Any]
    message: str

# --------- Helper Functions ---------

def _build_nomination_url(need_id: str, user_id: str) -> str:
    """Build the full nomination endpoint URL"""
    return f"{SERVE_BASE_URL}/api/v1/serve-fulfill/nomination/{need_id}/nominate/{user_id}"

def _create_error_response(
    need_id: str,
    user_id: str,
    status_code: int,
    error_message: str,
    raw_data: Optional[Dict[str, Any]] = None
) -> NominateResponse:
    """Create a standardized error response"""
    return NominateResponse(
        ok=False,
        needId=need_id,
        nominatedUserId=user_id,
        status_code=status_code,
        raw=raw_data or {"error": error_message},
        message=error_message
    )

def _create_success_response(
    need_id: str,
    user_id: str,
    status_code: int,
    response_data: Dict[str, Any]
) -> NominateResponse:
    """Create a standardized success response"""
    # Try to extract a meaningful message from the response
    message = "Nomination successful"
    if isinstance(response_data, dict):
        if "message" in response_data:
            message = str(response_data["message"])
        elif "status" in response_data:
            message = f"Nomination {response_data['status']}"
    
    return NominateResponse(
        ok=True,
        needId=need_id,
        nominatedUserId=user_id,
        status_code=status_code,
        raw=response_data,
        message=message
    )

# --------- Main Endpoint ---------

@router.post("/serve.fulfill.nominate", response_model=NominateResponse)
async def serve_fulfill_nominate(req: NominateRequest) -> NominateResponse:
    """
    Nominate a volunteer for a Serve need.
    
    Makes a POST request to the Serve API nomination endpoint.
    Returns normalized response with success/failure status.
    """
    # Build endpoint URL
    url = _build_nomination_url(req.needId, req.nominatedUserId)
    
    # Prepare headers
    headers = {
        "Accept": "application/json"
    }
    
    # Prepare query parameters for optional fields (API requires empty body)
    params: Dict[str, Any] = {}
    if req.source:
        params["source"] = req.source
    if req.idempotency_key:
        params["idempotency_key"] = req.idempotency_key
    
    # Make HTTP request
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # POST with empty body as per API spec, optional fields in query params
            response = await client.post(url, params=params if params else None, headers=headers)
            
            # Handle response
            status_code = response.status_code
            
            # Try to parse JSON response
            try:
                response_data = response.json() if response.content else {}
            except Exception:
                # If JSON parsing fails, use text content
                response_data = {"text": response.text}
            
            # Check if successful (2xx status codes)
            if 200 <= status_code < 300:
                return _create_success_response(
                    req.needId,
                    req.nominatedUserId,
                    status_code,
                    response_data
                )
            else:
                # Non-2xx response
                error_msg = f"Nomination failed: HTTP {status_code}"
                if isinstance(response_data, dict) and "message" in response_data:
                    error_msg = str(response_data["message"])
                elif isinstance(response_data, dict) and "error" in response_data:
                    error_msg = str(response_data["error"])
                elif response.text:
                    error_msg = f"Nomination failed: {response.text[:200]}"
                
                return _create_error_response(
                    req.needId,
                    req.nominatedUserId,
                    status_code,
                    error_msg,
                    response_data
                )
    
    except httpx.TimeoutException as e:
        error_msg = f"Request timeout: {str(e)}"
        return _create_error_response(
            req.needId,
            req.nominatedUserId,
            0,
            error_msg,
            {"timeout": True, "error": str(e)}
        )
    
    except httpx.ConnectError as e:
        error_msg = f"Connection error: {str(e)}"
        return _create_error_response(
            req.needId,
            req.nominatedUserId,
            0,
            error_msg,
            {"connection_error": True, "error": str(e)}
        )
    
    except httpx.HTTPError as e:
        error_msg = f"HTTP error: {str(e)}"
        return _create_error_response(
            req.needId,
            req.nominatedUserId,
            0,
            error_msg,
            {"http_error": True, "error": str(e)}
        )
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return _create_error_response(
            req.needId,
            req.nominatedUserId,
            0,
            error_msg,
            {"unexpected_error": True, "error": str(e)}
        )

