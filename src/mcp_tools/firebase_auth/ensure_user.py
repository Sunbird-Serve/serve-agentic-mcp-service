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
from firebase_admin import auth as firebase_auth
from firebase_admin.auth import UserNotFoundError, ActionCodeSettings

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
    reset_link: Optional[str] = Field(None, description="Password reset link (if generated)")
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

# --------- Main Endpoint ---------

@router.post("/firebase.auth.ensure_user", response_model=EnsureUserResponse)
async def ensure_user(req: EnsureUserRequest) -> EnsureUserResponse:
    """
    Idempotently ensure a Firebase email/password user exists.
    
    - Checks if user exists by email
    - If not exists and create_if_missing=True, creates user with random password
    - If generate_reset_link=True, generates password reset link
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
            
            # Generate reset link if requested
            reset_link = None
            if req.generate_reset_link:
                try:
                    # Get reset continue URL from config (if set)
                    continue_url = getattr(settings, 'FIREBASE_RESET_CONTINUE_URL', None)
                    
                    action_code_settings = None
                    if continue_url:
                        action_code_settings = ActionCodeSettings(
                            url=continue_url,
                            handle_code_in_app=False
                        )
                    
                    reset_link = auth_client.generate_password_reset_link(
                        req.email,
                        action_code_settings=action_code_settings
                    )
                    message += " (reset link generated)"
                except Exception as e:
                    # Log error but don't fail the request
                    return EnsureUserResponse(
                        status=status,
                        firebase_uid=firebase_uid,
                        message=message + f" (reset link generation failed: {str(e)})",
                        errors=[f"Reset link generation failed: {str(e)}"]
                    )
            
            return EnsureUserResponse(
                status=status,
                firebase_uid=firebase_uid,
                reset_link=reset_link,
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
        
        # Generate reset link if requested
        reset_link = None
        if req.generate_reset_link:
            try:
                # Get reset continue URL from config (if set)
                continue_url = getattr(settings, 'FIREBASE_RESET_CONTINUE_URL', None)
                
                action_code_settings = None
                if continue_url:
                    action_code_settings = ActionCodeSettings(
                        url=continue_url,
                        handle_code_in_app=False
                    )
                
                reset_link = auth_client.generate_password_reset_link(
                    req.email,
                    action_code_settings=action_code_settings
                )
                message += " (reset link generated)"
            except Exception as e:
                # Log error but don't fail the request
                return EnsureUserResponse(
                    status=status,
                    firebase_uid=firebase_uid,
                    message=message + f" (reset link generation failed: {str(e)})",
                    errors=[f"Reset link generation failed: {str(e)}"]
                )
        
        return EnsureUserResponse(
            status=status,
            firebase_uid=firebase_uid,
            reset_link=reset_link,
            message=message
        )
    
    except Exception as e:
        return EnsureUserResponse(
            status="failed",
            message=f"Error ensuring user: {str(e)}",
            errors=[str(e)]
        )

