"""
Firebase Auth: Ensure User Tool
- Idempotently ensure a Firebase email/password user exists
- Optionally generate password reset link
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
import secrets
import string
import httpx
from firebase_admin.auth import UserNotFoundError

from .firebase_client import get_firebase_auth, mask_email
from config import settings

router = APIRouter()

# --------- Models ---------

class EnsureUserRequest(BaseModel):
    """Request to ensure a Firebase user exists"""
    email: str = Field(..., description="Email address")
    display_name: Optional[str] = Field(None, description="Display name for the user")
    create_if_missing: bool = Field(True, description="Create user if doesn't exist")
    generate_reset_link: bool = Field(True, description="Generate password reset link")

class EnsureUserResponse(BaseModel):
    """Response from ensure user operation"""
    status: str = Field(..., description="Status: 'created', 'existing', or 'failed'")
    firebase_uid: Optional[str] = Field(None, description="Firebase User ID")
    reset_link: Optional[str] = Field(None, description="Password reset link (deprecated; not returned when using REST email)")
    reset_email_sent: bool = Field(False, description="Whether a reset email was requested from Firebase")
    message: str = Field("", description="Human-readable message")
    errors: List[str] = Field(default_factory=list, description="List of error messages if any")

# --------- Helper Functions ---------

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

def _generate_strong_password(length: int = 16) -> str:
    """
    Generate a strong random password.
    Includes uppercase, lowercase, digits, and special characters.
    """
    if length < 16:
        length = 16
    
    # Character sets
    uppercase = string.ascii_uppercase
    lowercase = string.ascii_lowercase
    digits = string.digits
    special = "!@#$%^&*"
    
    # Ensure at least one character from each set
    password = [
        secrets.choice(uppercase),
        secrets.choice(lowercase),
        secrets.choice(digits),
        secrets.choice(special)
    ]
    
    # Fill the rest with random characters from all sets
    all_chars = uppercase + lowercase + digits + special
    password.extend(secrets.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle to avoid predictable pattern
    secrets.SystemRandom().shuffle(password)
    
    return ''.join(password)

async def _send_password_reset_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Send password reset email via Firebase Auth REST API (sendOobCode).
    Returns (sent, error_message).
    """
    api_key = getattr(settings, "FIREBASE_WEB_API_KEY", None)
    if not api_key:
        return False, "FIREBASE_WEB_API_KEY not configured"

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}"
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": email.strip()
    }

    continue_url = getattr(settings, "FIREBASE_RESET_CONTINUE_URL", None)
    if continue_url:
        payload["continueUrl"] = continue_url

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                return True, None
            error_text = response.text
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", error_text)
            except Exception:
                error_msg = error_text
            return False, f"Reset email failed (HTTP {response.status_code}): {error_msg}"
    except httpx.TimeoutException:
        return False, "Reset email timeout"
    except Exception as e:
        return False, f"Reset email error: {str(e)}"

# --------- Main Endpoint ---------

@router.post("/firebase.auth.ensure_user", response_model=EnsureUserResponse)
async def ensure_user(req: EnsureUserRequest) -> EnsureUserResponse:
    """
    Idempotently ensure a Firebase email/password user exists.
    
    - Checks if user exists by email
    - If not exists and create_if_missing=True, creates user with random password
    - If generate_reset_link=True, sends password reset email via Firebase REST API
    """
    masked_email = mask_email(req.email)
    
    # Validate email format
    email_valid, email_error = _validate_email(req.email)
    if not email_valid:
        return EnsureUserResponse(
            status="failed",
            message="Invalid email format",
            errors=[email_error or "Invalid email format"]
        )
    
    try:
        # Get Firebase Auth client (initializes Firebase Admin SDK if needed)
        auth_client = get_firebase_auth()
        
        # Try to get user by email
        user = None
        user_exists = False
        
        try:
            user = auth_client.get_user_by_email(req.email)
            user_exists = True
        except UserNotFoundError:
            user_exists = False
        
        # If user exists, return existing status
        if user_exists and user:
            firebase_uid = user.uid
            status = "existing"
            message = f"User already exists (uid: {firebase_uid})"
            
            # Send reset email if requested
            reset_link = None
            reset_email_sent = False
            if req.generate_reset_link:
                sent, send_error = await _send_password_reset_email(req.email)
                reset_email_sent = sent
                if sent:
                    message += " (reset email sent)"
                else:
                    # Log error but don't fail the request
                    return EnsureUserResponse(
                        status=status,
                        firebase_uid=firebase_uid,
                        reset_email_sent=reset_email_sent,
                        message=message + f" (reset email failed: {send_error})",
                        errors=[f"Reset email failed: {send_error}"]
                    )
            
            return EnsureUserResponse(
                status=status,
                firebase_uid=firebase_uid,
                reset_link=reset_link,
                reset_email_sent=reset_email_sent,
                message=message
            )
        
        # User doesn't exist
        if not req.create_if_missing:
            return EnsureUserResponse(
                status="failed",
                message="User does not exist and create_if_missing is False",
                errors=["User not found and creation disabled"]
            )
        
        # Create user with random password
        generated_password = _generate_strong_password(16)
        
        create_user_params = {
            "email": req.email.strip(),
            "password": generated_password,
            "email_verified": False
        }
        
        if req.display_name:
            create_user_params["display_name"] = req.display_name.strip()
        
        new_user = auth_client.create_user(**create_user_params)
        firebase_uid = new_user.uid
        
        status = "created"
        message = f"User created successfully (uid: {firebase_uid})"
        
        # Send reset email if requested
        reset_link = None
        reset_email_sent = False
        if req.generate_reset_link:
            sent, send_error = await _send_password_reset_email(req.email)
            reset_email_sent = sent
            if sent:
                message += " (reset email sent)"
            else:
                # Log error but don't fail the request
                return EnsureUserResponse(
                    status=status,
                    firebase_uid=firebase_uid,
                    reset_email_sent=reset_email_sent,
                    message=message + f" (reset email failed: {send_error})",
                    errors=[f"Reset email failed: {send_error}"]
                )
        
        return EnsureUserResponse(
            status=status,
            firebase_uid=firebase_uid,
            reset_link=reset_link,
            reset_email_sent=reset_email_sent,
            message=message
        )
    
    except Exception as e:
        return EnsureUserResponse(
            status="failed",
            message=f"Error ensuring user: {str(e)}",
            errors=[str(e)]
        )

