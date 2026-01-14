"""
SERVE Volunteer: Update Status Tool
- Update volunteer lifecycle status in SERVE after Selection Agent decision
- Idempotent and safe to retry
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import httpx

from config import settings

router = APIRouter()

# --------- Constants ---------

SERVE_BASE_URL = settings.SERVE_BASE_URL
HTTP_TIMEOUT = 5.0  # seconds
MAX_RETRIES = 1  # Retry once on network failure

# --------- Models ---------

class UpdateStatusRequest(BaseModel):
    """Request to update volunteer status"""
    volunteer_id: str = Field(..., description="SERVE volunteer osid")
    status: Literal["RECOMMENDED", "ONHOLD"] = Field(..., description="New status: RECOMMENDED or ONHOLD")
    send: bool = Field(True, description="Send notification (default: true)")

class UpdateStatusResponse(BaseModel):
    """Response from status update"""
    status: str = Field(..., description="Status: 'success' or 'failed'")
    volunteer_id: str = Field(..., description="Volunteer ID that was updated")
    updated_status: Optional[Literal["RECOMMENDED", "ONHOLD"]] = Field(None, description="Updated status if successful")
    http_status: Optional[int] = Field(None, description="HTTP status code from SERVE API")
    message: str = Field("", description="Human-readable message")
    errors: List[str] = Field(default_factory=list, description="List of error messages if any")

# --------- Helper Functions ---------

def _mask_volunteer_id(volunteer_id: str) -> str:
    """Mask volunteer_id for logging: first 4 chars + ***"""
    if not volunteer_id:
        return "***"
    volunteer_id_str = str(volunteer_id).strip()
    if len(volunteer_id_str) <= 4:
        return "****"
    return volunteer_id_str[:4] + "***"

def _build_update_status_url(volunteer_id: str) -> str:
    """Build the URL for updating volunteer status"""
    return f"{SERVE_BASE_URL}/api/v1/serve-volunteering/user/status/update/{volunteer_id}"

# --------- Main Endpoint ---------

@router.post("/serve.volunteer.update_status", response_model=UpdateStatusResponse)
async def update_status(req: UpdateStatusRequest) -> UpdateStatusResponse:
    """
    Update SERVE volunteer lifecycle status after Selection Agent decision.
    
    This tool is idempotent and safe to retry. If the same status is sent
    multiple times, it will be treated as success.
    
    Status values:
    - RECOMMENDED: volunteer can proceed to fulfillment
    - ONHOLD: requires human follow-up / later decision
    """
    masked_volunteer_id = _mask_volunteer_id(req.volunteer_id)
    
    # Validate inputs
    if not req.volunteer_id or not req.volunteer_id.strip():
        return UpdateStatusResponse(
            status="failed",
            volunteer_id=req.volunteer_id or "",
            message="Validation failed: volunteer_id is required",
            errors=["volunteer_id must be non-empty"]
        )
    
    volunteer_id = req.volunteer_id.strip()
    
    # Build URL
    url = _build_update_status_url(volunteer_id)
    
    # Build request body
    payload = {
        "status": req.status,
        "send": req.send
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Retry logic for network failures
    last_error = None
    last_http_status = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.put(url, json=payload, headers=headers)
                
                http_status = response.status_code
                last_http_status = http_status
                
                # Handle 2xx success
                if 200 <= http_status < 300:
                    return UpdateStatusResponse(
                        status="success",
                        volunteer_id=volunteer_id,
                        updated_status=req.status,
                        http_status=http_status,
                        message=f"Status updated successfully to {req.status}"
                    )
                
                # Handle 4xx client errors (do not retry)
                if 400 <= http_status < 500:
                    error_message = f"HTTP {http_status}"
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('message') or error_data.get('error', {}).get('message', '')
                        if error_msg:
                            error_message = error_msg
                    except:
                        if response.text:
                            error_message = response.text[:200]
                    
                    # Check if it's a duplicate/conflict (same status) - treat as success for idempotency
                    if http_status == 409 or "already" in error_message.lower() or "duplicate" in error_message.lower():
                        return UpdateStatusResponse(
                            status="success",
                            volunteer_id=volunteer_id,
                            updated_status=req.status,
                            http_status=http_status,
                            message=f"Status already set to {req.status} (idempotent success)"
                        )
                    
                    return UpdateStatusResponse(
                        status="failed",
                        volunteer_id=volunteer_id,
                        http_status=http_status,
                        message=f"Client error: {error_message}",
                        errors=[error_message]
                    )
                
                # Handle 5xx server errors (retry on next iteration if attempts remain)
                if 500 <= http_status < 600:
                    last_error = f"Server error: HTTP {http_status}"
                    if attempt < MAX_RETRIES:
                        continue  # Retry
                    else:
                        # Last attempt failed
                        return UpdateStatusResponse(
                            status="failed",
                            volunteer_id=volunteer_id,
                            http_status=http_status,
                            message=last_error,
                            errors=[last_error]
                        )
                
                # Other status codes
                error_text = response.text[:200] if response.text else ""
                last_error = f"Unexpected HTTP {http_status}: {error_text}"
                if attempt < MAX_RETRIES:
                    continue  # Retry
                else:
                    return UpdateStatusResponse(
                        status="failed",
                        volunteer_id=volunteer_id,
                        http_status=http_status,
                        message=last_error,
                        errors=[last_error]
                    )
        
        except httpx.TimeoutException:
            last_error = "Request timeout"
            if attempt < MAX_RETRIES:
                continue  # Retry
            else:
                return UpdateStatusResponse(
                    status="failed",
                    volunteer_id=volunteer_id,
                    message="Request timeout after retries",
                    errors=["Request timeout"]
                )
        
        except httpx.ConnectError as e:
            last_error = f"Connection error: {str(e)}"
            if attempt < MAX_RETRIES:
                continue  # Retry
            else:
                return UpdateStatusResponse(
                    status="failed",
                    volunteer_id=volunteer_id,
                    message="Connection error after retries",
                    errors=[last_error]
                )
        
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            if attempt < MAX_RETRIES:
                continue  # Retry
            else:
                return UpdateStatusResponse(
                    status="failed",
                    volunteer_id=volunteer_id,
                    message=last_error,
                    errors=[last_error]
                )
    
    # Should not reach here, but handle just in case
    return UpdateStatusResponse(
        status="failed",
        volunteer_id=volunteer_id,
        http_status=last_http_status,
        message=last_error or "Unknown error",
        errors=[last_error or "Unknown error"]
    )

