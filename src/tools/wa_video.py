"""
WhatsApp Video Tool
- Send in-app playable WhatsApp video messages
- Uploads video to WhatsApp Media API and sends using media_id
- Implements caching to avoid re-uploading the same file
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Literal, Tuple
import hashlib
import os

from config import settings
from tools.wa_media import upload_media_to_whatsapp, send_video_message
from tools.wa_media_cache import get_cached_media_id, save_cached_media_id

router = APIRouter()

# --------- Models ---------

class SendClassVideoRequest(BaseModel):
    """Request to send class video via WhatsApp"""
    to_phone: str = Field(..., description="Recipient phone number (E.164 format)")
    caption: Optional[str] = Field(None, description="Optional caption text for the video")

class SendClassVideoResponse(BaseModel):
    """Response from sending class video"""
    ok: bool
    media_id: Optional[str] = None
    wa_message_id: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None

# --------- Helper Functions ---------

def _calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculate SHA256 hash of a file"""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def _resolve_video_path(config_path: str, config_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve video file path from configuration.
    
    Args:
        config_path: Path from settings (e.g., settings.SERVE_CLASS_VIDEO_PATH)
        config_name: Name of config variable for error messages
    
    Returns:
        Tuple of (file_path, error_message)
    """
    if not config_path:
        return None, f"{config_name} not configured"
    
    # Resolve path (supports relative and absolute paths)
    resolved_path = os.path.abspath(config_path)
    if not os.path.exists(resolved_path):
        return None, f"Video file not found: {resolved_path}. Please place the video file at the configured path."
    
    if not os.path.isfile(resolved_path):
        return None, f"Path is not a file: {resolved_path}"
    
    return resolved_path, None

async def _send_video_internal(
    video_path_config: str,
    config_name: str,
    to_phone: str,
    caption: Optional[str]
) -> SendClassVideoResponse:
    """
    Internal function to send a video via WhatsApp.
    Handles validation, caching, upload, and sending.
    
    Args:
        video_path_config: Video path from settings
        config_name: Name of config variable for error messages
        to_phone: Recipient phone number
        caption: Optional caption text
    
    Returns:
        SendClassVideoResponse with result
    """
    # Validate configuration
    if not settings.WHATSAPP_ACCESS_TOKEN:
        return SendClassVideoResponse(
            ok=False,
            error="WHATSAPP_ACCESS_TOKEN not configured"
        )
    
    if not settings.WHATSAPP_PHONE_NUMBER_ID:
        return SendClassVideoResponse(
            ok=False,
            error="WHATSAPP_PHONE_NUMBER_ID not configured"
        )
    
    # Resolve video file path from config
    video_path, path_error = _resolve_video_path(video_path_config, config_name)
    if path_error:
        return SendClassVideoResponse(
            ok=False,
            error=path_error
        )
    
    # Calculate file hash for caching
    file_hash = _calculate_file_hash(video_path)
    if not file_hash:
        return SendClassVideoResponse(
            ok=False,
            error="Failed to calculate file hash"
        )
    
    # Check cache for existing media_id
    cached_media_id = get_cached_media_id(
        file_hash,
        settings.WHATSAPP_PHONE_NUMBER_ID,
        settings.WA_MEDIA_CACHE_PATH
    )
    
    media_id = cached_media_id
    cached = cached_media_id is not None
    
    # Upload if not cached
    if not media_id:
        media_id, upload_error = await upload_media_to_whatsapp(
            file_path=video_path,
            mime_type="video/mp4",
            access_token=settings.WHATSAPP_ACCESS_TOKEN,
            phone_number_id=settings.WHATSAPP_PHONE_NUMBER_ID,
            api_version=settings.WHATSAPP_API_VERSION
        )
        
        if upload_error:
            return SendClassVideoResponse(
                ok=False,
                error=f"Upload failed: {upload_error}"
            )
        
        # Save to cache
        if media_id:
            save_cached_media_id(
                file_hash,
                settings.WHATSAPP_PHONE_NUMBER_ID,
                media_id,
                settings.WA_MEDIA_CACHE_PATH
            )
    
    # Send video message
    wa_message_id, send_error = await send_video_message(
        to_phone=to_phone,
        media_id=media_id,
        caption=caption,
        access_token=settings.WHATSAPP_ACCESS_TOKEN,
        phone_number_id=settings.WHATSAPP_PHONE_NUMBER_ID,
        api_version=settings.WHATSAPP_API_VERSION
    )
    
    if send_error:
        return SendClassVideoResponse(
            ok=False,
            media_id=media_id,
            cached=cached,
            error=f"Send failed: {send_error}"
        )
    
    # Success
    return SendClassVideoResponse(
        ok=True,
        media_id=media_id,
        wa_message_id=wa_message_id,
        cached=cached
    )

# --------- Main Endpoint ---------

@router.post("/serve.whatsapp.send_class_video", response_model=SendClassVideoResponse)
async def send_class_video(req: SendClassVideoRequest) -> SendClassVideoResponse:
    """
    Send a class demo video via WhatsApp.
    
    Uploads the video to WhatsApp Media API (with caching) and sends it
    as an in-app playable video message (not a link).
    """
    return await _send_video_internal(
        settings.SERVE_CLASS_VIDEO_PATH,
        "SERVE_CLASS_VIDEO_PATH",
        req.to_phone,
        req.caption
    )

@router.post("/serve.whatsapp.send_welcome_video", response_model=SendClassVideoResponse)
async def send_welcome_video(req: SendClassVideoRequest) -> SendClassVideoResponse:
    """
    Send a welcome video via WhatsApp.
    
    Uploads the video to WhatsApp Media API (with caching) and sends it
    as an in-app playable video message (not a link).
    """
    return await _send_video_internal(
        settings.SERVE_WELCOME_VIDEO_PATH,
        "SERVE_WELCOME_VIDEO_PATH",
        req.to_phone,
        req.caption
    )

