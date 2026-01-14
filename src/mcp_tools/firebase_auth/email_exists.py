"""
Firebase Auth: Email Exists Tool
- Check if a Firebase user exists by email
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from firebase_admin import auth as firebase_auth
from firebase_admin.auth import UserNotFoundError

from .firebase_client import get_firebase_auth, mask_email

router = APIRouter()

# --------- Models ---------

class EmailExistsRequest(BaseModel):
    """Request to check if Firebase user email exists"""
    email: str = Field(..., description="Email address to check")

class EmailExistsResponse(BaseModel):
    """Response from email existence check"""
    status: str = Field(..., description="Status: 'ok' or 'error'")
    exists: bool = Field(..., description="Whether the email exists in Firebase")
    firebase_uid: Optional[str] = Field(None, description="Firebase User ID if found")
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

# --------- Main Endpoint ---------

@router.post("/firebase.auth.email_exists", response_model=EmailExistsResponse)
async def email_exists(req: EmailExistsRequest) -> EmailExistsResponse:
    """
    Check if a Firebase user exists by email address.
    
    Returns Firebase UID if the user exists.
    """
    masked_email = mask_email(req.email)
    
    # Validate email format
    email_valid, email_error = _validate_email(req.email)
    if not email_valid:
        return EmailExistsResponse(
            status="error",
            exists=False,
            message="Invalid email format",
            errors=[email_error or "Invalid email format"]
        )
    
    try:
        # Get Firebase Auth client (initializes Firebase Admin SDK if needed)
        auth_client = get_firebase_auth()
        
        # Try to get user by email
        user = auth_client.get_user_by_email(req.email)
        
        # User found
        return EmailExistsResponse(
            status="ok",
            exists=True,
            firebase_uid=user.uid,
            message=f"Email exists in Firebase (uid: {user.uid})"
        )
    
    except UserNotFoundError:
        # User not found - this is a valid response, not an error
        return EmailExistsResponse(
            status="ok",
            exists=False,
            message="Email not found in Firebase"
        )
    
    except Exception as e:
        # Other errors
        return EmailExistsResponse(
            status="error",
            exists=False,
            message=f"Error checking email: {str(e)}",
            errors=[str(e)]
        )

