"""
Serve Volunteering Tools
- Check if volunteer email exists in SERVE system
- Get volunteer ID if exists
- Register new volunteer in SERVE system
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
import httpx
from datetime import datetime

from config import settings

router = APIRouter()

# --------- Constants ---------

SERVE_BASE_URL = settings.SERVE_BASE_URL
SERVE_TIMEOUT = settings.SERVE_TIMEOUT_SECONDS
SERVE_DEFAULT_AGENCY_ID = settings.SERVE_DEFAULT_AGENCY_ID

# --------- Models ---------

class EmailExistsRequest(BaseModel):
    """Request to check if volunteer email exists"""
    email: str = Field(..., description="Email address to check")

class EmailExistsResponse(BaseModel):
    """Response from email existence check"""
    status: str = Field(..., description="Status: 'ok', 'not_found', or 'error'")
    exists: bool = Field(..., description="Whether the email exists in SERVE")
    volunteer_id: Optional[str] = Field(None, description="Volunteer ID (osid) if found")
    message: str = Field("", description="Human-readable message")
    errors: List[str] = Field(default_factory=list, description="List of error messages if any")

class RegisterVolunteerRequest(BaseModel):
    """Request to register a new volunteer"""
    name: str = Field(..., description="Full name of the volunteer")
    email: str = Field(..., description="Email address")
    wa_phone: str = Field(..., description="WhatsApp phone number with country code")
    day_preferred: Optional[List[str]] = Field(None, description="Preferred days, e.g. ['Monday', 'Wednesday']")
    time_preferred: Optional[List[str]] = Field(None, description="Preferred times, e.g. ['Morning', 'Afternoon']")
    agency_id: Optional[str] = Field(None, description="Agency ID (uses default if not provided)")
    idempotency_key: str = Field(..., description="Idempotency key for safe retries")

class RegisterVolunteerResponse(BaseModel):
    """Response from volunteer registration"""
    status: str = Field(..., description="Status: 'created' or 'failed'")
    volunteer_id: Optional[str] = Field(None, description="Volunteer ID (osid) from API-1")
    message: str = Field("", description="Human-readable message")
    errors: List[str] = Field(default_factory=list, description="List of error messages if any")

# --------- Helper Functions ---------

def _mask_email(email: str) -> str:
    """Mask email for logging: first 2 chars + '***@' + domain"""
    if not email or '@' not in email:
        return "***"
    
    parts = email.split('@', 1)
    if len(parts) != 2:
        return "***"
    
    local_part, domain = parts
    if len(local_part) <= 2:
        masked_local = "***"
    else:
        masked_local = local_part[:2] + "***"
    
    return f"{masked_local}@{domain}"

def _mask_phone(phone: str) -> str:
    """Mask phone for logging: show only last 4 digits"""
    if not phone:
        return "***"
    phone_str = str(phone).strip()
    if len(phone_str) <= 4:
        return "****"
    return "****" + phone_str[-4:]

def _validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """Basic email format validation. Returns (is_valid, error_message)"""
    if not email or not email.strip():
        return False, "Email is required"
    
    email = email.strip()
    
    # Basic format check: must contain @ and have at least one char before and after @
    if '@' not in email:
        return False, "Email must contain @"
    
    parts = email.split('@')
    if len(parts) != 2:
        return False, "Email must have exactly one @"
    
    local_part, domain = parts
    if not local_part or not local_part.strip():
        return False, "Email must have characters before @"
    
    if not domain or not domain.strip() or '.' not in domain:
        return False, "Email must have a valid domain"
    
    return True, None

def _extract_volunteer_id(response_data: dict) -> Optional[str]:
    """
    Extract volunteer ID (osid) from response.
    Checks multiple possible locations in the response structure.
    """
    if not isinstance(response_data, dict):
        return None
    
    # Try different possible paths
    paths_to_try = [
        ["result", "Users", "osid"],
        ["result", "user", "osid"],
        ["result", "osid"],
        ["osid"],
        ["data", "osid"],
        ["user", "osid"],
        ["Users", "osid"]
    ]
    
    for path in paths_to_try:
        value = response_data
        try:
            for key in path:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    break
            else:
                if value and isinstance(value, str):
                    return value
        except (KeyError, TypeError, AttributeError):
            continue
    
    return None

# --------- Main Endpoint ---------

@router.post("/serve.volunteer.email_exists", response_model=EmailExistsResponse)
async def email_exists(req: EmailExistsRequest) -> EmailExistsResponse:
    """
    Check if a volunteer email exists in SERVE system.
    
    Returns volunteer ID (osid) if the email exists.
    """
    masked_email = _mask_email(req.email)
    url = f"{SERVE_BASE_URL}/api/v1/serve-volunteering/user/email"
    
    params = {
        "email": req.email
    }
    
    try:
        async with httpx.AsyncClient(timeout=SERVE_TIMEOUT) as client:
            response = await client.get(url, params=params)
            
            status_code = response.status_code
            
            # Handle 200 OK - email exists
            if status_code == 200:
                try:
                    response_data = response.json()
                except Exception as e:
                    return EmailExistsResponse(
                        status="error",
                        exists=False,
                        message=f"Failed to parse response JSON: {str(e)}",
                        errors=[f"JSON parse error: {str(e)}"]
                    )
                
                # Try to extract volunteer_id
                volunteer_id = _extract_volunteer_id(response_data)
                
                return EmailExistsResponse(
                    status="ok",
                    exists=True,
                    volunteer_id=volunteer_id,
                    message="Email exists in SERVE" + (f" (volunteer_id: {volunteer_id})" if volunteer_id else "")
                )
            
            # Handle 404 Not Found - email doesn't exist
            elif status_code == 404:
                return EmailExistsResponse(
                    status="not_found",
                    exists=False,
                    message="Email not found in SERVE system"
                )
            
            # Handle 204 No Content - email doesn't exist
            elif status_code == 204:
                return EmailExistsResponse(
                    status="not_found",
                    exists=False,
                    message="Email not found in SERVE system"
                )
            
            # Handle 400 Bad Request - might be invalid email
            elif status_code == 400:
                error_message = "Invalid email or bad request"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message') or error_data.get('error', {}).get('message', '')
                    if error_msg:
                        error_message = error_msg
                except:
                    error_message = response.text[:200] if response.text else "Bad request"
                
                # Check if it's specifically an invalid email error
                if "invalid" in error_message.lower() or "email" in error_message.lower():
                    return EmailExistsResponse(
                        status="error",
                        exists=False,
                        message="Invalid email format",
                        errors=[error_message]
                    )
                
                return EmailExistsResponse(
                    status="error",
                    exists=False,
                    message=f"Bad request: {error_message}",
                    errors=[error_message]
                )
            
            # Handle other errors
            else:
                error_message = f"HTTP {status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message') or error_data.get('error', {}).get('message', '')
                    if error_msg:
                        error_message = error_msg
                except:
                    if response.text:
                        error_message = response.text[:200]
                
                return EmailExistsResponse(
                    status="error",
                    exists=False,
                    message=f"Request failed: {error_message}",
                    errors=[f"HTTP {status_code}: {error_message}"]
                )
    
    except httpx.TimeoutException:
        return EmailExistsResponse(
            status="error",
            exists=False,
            message="Request timeout",
            errors=["Request to SERVE API timed out"]
        )
    
    except httpx.ConnectError as e:
        return EmailExistsResponse(
            status="error",
            exists=False,
            message="Connection error",
            errors=[f"Failed to connect to SERVE API: {str(e)}"]
        )
    
    except Exception as e:
        return EmailExistsResponse(
            status="error",
            exists=False,
            message=f"Unexpected error: {str(e)}",
            errors=[str(e)]
        )

@router.post("/serve.volunteer.register", response_model=RegisterVolunteerResponse)
async def register_volunteer(req: RegisterVolunteerRequest) -> RegisterVolunteerResponse:
    """
    Register a new volunteer in SERVE system.
    
    Calls two APIs in sequence:
    1. Create User (POST /user/)
    2. Create User Profile (POST /user/user-profile)
    
    This tool should only be called when email_exists returned exists=false.
    """
    masked_email = _mask_email(req.email)
    masked_phone = _mask_phone(req.wa_phone)
    
    # Validate inputs
    errors = []
    
    # Validate name
    if not req.name or not req.name.strip():
        errors.append("Name is required and cannot be empty")
    
    # Validate email
    email_valid, email_error = _validate_email(req.email)
    if not email_valid:
        errors.append(email_error or "Invalid email format")
    
    # Validate phone
    if not req.wa_phone or not req.wa_phone.strip():
        errors.append("Phone number is required and cannot be empty")
    
    if errors:
        return RegisterVolunteerResponse(
            status="failed",
            message="Validation failed",
            errors=errors
        )
    
    # Use default agency_id if not provided
    agency_id = req.agency_id if req.agency_id else SERVE_DEFAULT_AGENCY_ID
    
    # Get today's date in YYYY-MM-DD format
    today = datetime.now().strftime("%Y-%m-%d")
    
    # API-1: Create User
    create_user_url = f"{SERVE_BASE_URL}/api/v1/serve-volunteering/user/"
    
    create_user_payload = {
        "identityDetails": {
            "fullname": req.name.strip(),
            "name": req.name.strip(),
            "gender": "Female",
            "dob": "1990-01-01",
            "Nationality": "India"
        },
        "contactDetails": {
            "email": req.email.strip(),
            "mobile": req.wa_phone.strip(),
            "address": {
                "city": "",
                "state": "",
                "country": "India"
            }
        },
        "agencyId": agency_id,
        "status": "Registered",
        "role": ["Volunteer"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    volunteer_id = None
    
    try:
        async with httpx.AsyncClient(timeout=SERVE_TIMEOUT) as client:
            # API-1: Create User
            create_user_response = await client.post(
                create_user_url,
                json=create_user_payload,
                headers=headers
            )
            
            if create_user_response.status_code not in (200, 201):
                error_text = create_user_response.text[:500] if create_user_response.text else ""
                try:
                    error_data = create_user_response.json()
                    error_msg = error_data.get('message') or error_data.get('error', {}).get('message', error_text)
                except:
                    error_msg = error_text
                
                return RegisterVolunteerResponse(
                    status="failed",
                    message=f"Failed to create user: HTTP {create_user_response.status_code}",
                    errors=[f"Create user failed (HTTP {create_user_response.status_code}): {error_msg}"]
                )
            
            # Parse volunteer_id from response
            try:
                user_response_data = create_user_response.json()
                # Try to extract osid from result.Users.osid
                volunteer_id = (
                    user_response_data.get("result", {})
                    .get("Users", {})
                    .get("osid")
                )
                
                if not volunteer_id:
                    # Try alternative paths
                    volunteer_id = user_response_data.get("result", {}).get("osid")
                    if not volunteer_id:
                        volunteer_id = user_response_data.get("osid")
                
                if not volunteer_id:
                    error_snippet = str(user_response_data)[:200]
                    return RegisterVolunteerResponse(
                        status="failed",
                        message="Failed to extract volunteer_id from create user response",
                        errors=[f"No osid found in response: {error_snippet}"]
                    )
            
            except Exception as e:
                return RegisterVolunteerResponse(
                    status="failed",
                    message=f"Failed to parse create user response: {str(e)}",
                    errors=[f"Response parse error: {str(e)}"]
                )
            
            # API-2: Create User Profile
            create_profile_url = f"{SERVE_BASE_URL}/api/v1/serve-volunteering/user/user-profile"
            
            create_profile_payload = {
                "skills": [],
                "genericDetails": {
                    "qualification": "Graduate",
                    "affiliation": "SERVE Volunteer",
                    "yearsOfExperience": "",
                    "employmentStatus": "Others"
                },
                "userPreference": {
                    "timePreferred": req.time_preferred or [],
                    "dayPreferred": req.day_preferred or [],
                    "interestArea": [],
                    "language": []
                },
                "agencyId": SERVE_DEFAULT_AGENCY_ID,
                "userId": volunteer_id,
                "onboardDetails": {
                    "onboardStatus": [{"onboardStep": "Discussion", "status": "completed"}],
                    "refreshPeriod": "2 years",
                    "profileCompletion": "50"
                },
                "consentDetails": {
                    "consentGiven": True,
                    "consentDate": today,
                    "consentDescription": "Consent given for sharing preference to other volunteer agency through secure network"
                },
                "referenceChannelId": "",
                "volunteeringHours": {
                    "totalHours": 0,
                    "hoursPerWeek": 0
                }
            }
            
            create_profile_response = await client.post(
                create_profile_url,
                json=create_profile_payload,
                headers=headers
            )
            
            if create_profile_response.status_code not in (200, 201):
                error_text = create_profile_response.text[:500] if create_profile_response.text else ""
                try:
                    error_data = create_profile_response.json()
                    error_msg = error_data.get('message') or error_data.get('error', {}).get('message', error_text)
                except:
                    error_msg = error_text
                
                return RegisterVolunteerResponse(
                    status="failed",
                    volunteer_id=volunteer_id,
                    message=f"Failed to create user profile: HTTP {create_profile_response.status_code}",
                    errors=[f"Create profile failed (HTTP {create_profile_response.status_code}): {error_msg}"]
                )
            
            # Success
            return RegisterVolunteerResponse(
                status="created",
                volunteer_id=volunteer_id,
                message=f"Volunteer registered successfully (volunteer_id: {volunteer_id})"
            )
    
    except httpx.TimeoutException:
        return RegisterVolunteerResponse(
            status="failed",
            volunteer_id=volunteer_id,
            message="Request timeout",
            errors=["Request to SERVE API timed out"]
        )
    
    except httpx.ConnectError as e:
        return RegisterVolunteerResponse(
            status="failed",
            volunteer_id=volunteer_id,
            message="Connection error",
            errors=[f"Failed to connect to SERVE API: {str(e)}"]
        )
    
    except Exception as e:
        return RegisterVolunteerResponse(
            status="failed",
            volunteer_id=volunteer_id,
            message=f"Unexpected error: {str(e)}",
            errors=[str(e)]
        )

