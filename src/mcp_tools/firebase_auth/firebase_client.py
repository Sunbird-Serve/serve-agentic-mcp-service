"""
Firebase Admin SDK Client
- Singleton initialization
- Shared Firebase Auth client
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from typing import Optional

# Global variable to track if Firebase is initialized
_firebase_initialized = False

def get_firebase_auth() -> auth.Client:
    """
    Get Firebase Auth client, initializing Firebase Admin SDK if needed.
    Ensures singleton initialization (only initializes once per process).
    
    Returns:
        firebase_admin.auth.Client instance
    """
    global _firebase_initialized
    
    if _firebase_initialized:
        return auth
    
    # Initialize Firebase Admin SDK
    from config import settings
    
    service_account_path_or_json = settings.FIREBASE_SERVICE_ACCOUNT_JSON
    
    if not service_account_path_or_json:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON not configured")
    
    # Check if it's a file path or JSON string
    cred = None
    try:
        # Try as file path first
        if os.path.exists(service_account_path_or_json):
            cred = credentials.Certificate(service_account_path_or_json)
        else:
            # Try as JSON string
            try:
                service_account_dict = json.loads(service_account_path_or_json)
                cred = credentials.Certificate(service_account_dict)
            except json.JSONDecodeError:
                raise ValueError(f"FIREBASE_SERVICE_ACCOUNT_JSON is neither a valid file path nor valid JSON: {service_account_path_or_json[:50]}...")
    except Exception as e:
        raise ValueError(f"Failed to load Firebase service account: {str(e)}")
    
    # Initialize Firebase Admin SDK
    try:
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
    except ValueError:
        # App already initialized (can happen in some edge cases)
        _firebase_initialized = True
    
    return auth

def mask_email(email: str) -> str:
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

