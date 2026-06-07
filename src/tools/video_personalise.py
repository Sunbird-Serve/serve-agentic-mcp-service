"""
Video MCP: Personalised Welcome/Orientation Video
- Generate personalised video context for volunteers
- Send template video with personalised caption via WhatsApp
- Extensible architecture for future video overlay/TTS generation

Current implementation: Template video + personalised caption
Future extension points: ffmpeg overlay, TTS audio, external video API
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone
import os
import hashlib

from config import settings

router = APIRouter()

# --------- Configuration ---------

GENERATED_VIDEO_DIR = os.environ.get("GENERATED_VIDEO_DIR", "./generated_videos")

# --------- Video Templates ---------

VIDEO_TEMPLATES = {
    "welcome": {
        "path_setting": "SERVE_WELCOME_VIDEO_PATH",
        "default_caption": "Welcome to SERVE, {name}! We're excited to have you join our community of volunteers making a difference in education.",
    },
    "orientation": {
        "path_setting": "SERVE_CLASS_VIDEO_PATH",
        "default_caption": "Hi {name}! Here's a quick look at how orientation sessions work at SERVE. Watch this short video to know what to expect.",
    },
    "thankyou": {
        "path_setting": "SERVE_THANKYOU_VIDEO_PATH",
        "default_caption": "Thank you, {name}! Your commitment to volunteering is truly appreciated. Together we're making education accessible for all.",
    },
}

# --------- Models ---------


class GenerateVideoRequest(BaseModel):
    name: str = Field(..., description="Volunteer's name for personalisation")
    context: Literal["welcome", "orientation", "thankyou"] = Field(
        ..., description="Video context: 'welcome', 'orientation', or 'thankyou'"
    )
    locale: str = Field(default="en-IN", description="Locale for caption language")
    custom_message: Optional[str] = Field(
        None, description="Optional custom message to append to caption"
    )


class GenerateVideoResponse(BaseModel):
    ok: bool
    videoType: str
    videoPath: str = ""
    caption: str = ""
    personalisedFor: str = ""
    videoUrl: Optional[str] = None
    message: str = ""


class SendPersonalisedVideoRequest(BaseModel):
    to_phone: str = Field(..., description="Recipient phone number (E.164 format)")
    name: str = Field(..., description="Volunteer's name for personalisation")
    context: Literal["welcome", "orientation", "thankyou"] = Field(
        default="welcome", description="Video context"
    )
    custom_message: Optional[str] = Field(None, description="Optional custom caption text")


class SendPersonalisedVideoResponse(BaseModel):
    ok: bool
    media_id: Optional[str] = None
    wa_message_id: Optional[str] = None
    caption: str = ""
    error: Optional[str] = None


# --------- Helpers ---------


def _generate_caption(name: str, context: str, custom_message: Optional[str] = None, locale: str = "en-IN") -> str:
    """Generate a personalised caption for the video."""
    template_info = VIDEO_TEMPLATES.get(context, VIDEO_TEMPLATES["welcome"])
    caption = template_info["default_caption"].format(name=name)

    if custom_message:
        caption = f"{caption}\n\n{custom_message}"

    return caption


def _get_video_path(context: str) -> Optional[str]:
    """Get the video file path for a given context."""
    template_info = VIDEO_TEMPLATES.get(context)
    if not template_info:
        return None

    setting_name = template_info["path_setting"]
    path = getattr(settings, setting_name, "")
    if not path:
        return None

    resolved = os.path.abspath(path)
    if os.path.exists(resolved):
        return resolved
    return None


# --------- Endpoints ---------


@router.post("/video.generate_personalised", response_model=GenerateVideoResponse)
async def generate_personalised_video(req: GenerateVideoRequest) -> GenerateVideoResponse:
    """
    Generate a personalised video context for a volunteer.

    Currently returns the template video path with a personalised caption.
    The agent can then use this to send via WhatsApp or other channels.

    Extension point: When video generation infrastructure is available (ffmpeg,
    TTS API, or external video service), this endpoint will generate actual
    personalised video files with name overlay and/or TTS narration.
    """
    if req.context not in VIDEO_TEMPLATES:
        return GenerateVideoResponse(
            ok=False,
            videoType=req.context,
            message=f"Unknown video context: {req.context}. Valid: {list(VIDEO_TEMPLATES.keys())}"
        )

    video_path = _get_video_path(req.context)
    caption = _generate_caption(req.name, req.context, req.custom_message, req.locale)

    if not video_path:
        return GenerateVideoResponse(
            ok=False,
            videoType=req.context,
            caption=caption,
            personalisedFor=req.name,
            message=f"Video file not found for context '{req.context}'. Caption was generated successfully."
        )

    return GenerateVideoResponse(
        ok=True,
        videoType=req.context,
        videoPath=video_path,
        caption=caption,
        personalisedFor=req.name,
        message="Personalised video ready to send"
    )


@router.post("/video.send_personalised", response_model=SendPersonalisedVideoResponse)
async def send_personalised_video(req: SendPersonalisedVideoRequest) -> SendPersonalisedVideoResponse:
    """
    Generate and send a personalised video via WhatsApp in one step.

    Combines video generation (personalised caption) with WhatsApp delivery.
    Uses the existing wa_video infrastructure for upload and sending.
    """
    from tools.wa_video import _send_video_internal

    if not settings.WHATSAPP_ACCESS_TOKEN or not settings.WHATSAPP_PHONE_NUMBER_ID:
        return SendPersonalisedVideoResponse(
            ok=False,
            error="WhatsApp API not configured (missing ACCESS_TOKEN or PHONE_NUMBER_ID)"
        )

    caption = _generate_caption(req.name, req.context, req.custom_message)

    # Map context to settings path
    path_map = {
        "welcome": settings.SERVE_WELCOME_VIDEO_PATH,
        "orientation": settings.SERVE_CLASS_VIDEO_PATH,
        "thankyou": settings.SERVE_THANKYOU_VIDEO_PATH,
    }
    config_name_map = {
        "welcome": "SERVE_WELCOME_VIDEO_PATH",
        "orientation": "SERVE_CLASS_VIDEO_PATH",
        "thankyou": "SERVE_THANKYOU_VIDEO_PATH",
    }

    video_path = path_map.get(req.context, settings.SERVE_WELCOME_VIDEO_PATH)
    config_name = config_name_map.get(req.context, "SERVE_WELCOME_VIDEO_PATH")

    result = await _send_video_internal(video_path, config_name, req.to_phone, caption)

    return SendPersonalisedVideoResponse(
        ok=result.ok,
        media_id=result.media_id,
        wa_message_id=result.wa_message_id,
        caption=caption,
        error=result.error,
    )
