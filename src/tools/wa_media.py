"""
WhatsApp Media API Client
- Upload media files to WhatsApp Cloud API
- Send video messages using media_id
"""
import httpx
from typing import Optional, Tuple
import os

async def upload_media_to_whatsapp(
    file_path: str,
    mime_type: str,
    access_token: str,
    phone_number_id: str,
    api_version: str = "v21.0"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Upload a media file to WhatsApp Cloud API and get media_id.
    
    Args:
        file_path: Path to the media file
        mime_type: MIME type (e.g., "video/mp4")
        access_token: WhatsApp API access token
        phone_number_id: WhatsApp Business Phone Number ID
        api_version: WhatsApp API version (default: v21.0)
    
    Returns:
        Tuple of (media_id, error_message)
        If successful: (media_id, None)
        If failed: (None, error_message)
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    
    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/media"
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        # Read file and prepare multipart form data
        with open(file_path, 'rb') as f:
            files = {
                "file": (os.path.basename(file_path), f, mime_type),
                "type": (None, mime_type),
                "messaging_product": (None, "whatsapp")
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    media_id = data.get('id')
                    if media_id:
                        return media_id, None
                    else:
                        return None, f"No media_id in response: {data}"
                else:
                    error_text = response.text
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', {}).get('message', error_text)
                    except:
                        error_msg = error_text
                    return None, f"Upload failed (HTTP {response.status_code}): {error_msg}"
    
    except httpx.TimeoutException:
        return None, "Upload timeout"
    except Exception as e:
        return None, f"Upload error: {str(e)}"

async def send_video_message(
    to_phone: str,
    media_id: str,
    caption: Optional[str],
    access_token: str,
    phone_number_id: str,
    api_version: str = "v21.0"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Send a WhatsApp video message using media_id.
    
    Args:
        to_phone: Recipient phone number (E.164 format, with or without +)
        media_id: Media ID from upload_media_to_whatsapp
        caption: Optional caption text
        access_token: WhatsApp API access token
        phone_number_id: WhatsApp Business Phone Number ID
        api_version: WhatsApp API version (default: v21.0)
    
    Returns:
        Tuple of (wa_message_id, error_message)
        If successful: (wa_message_id, None)
        If failed: (None, error_message)
    """
    # Normalize phone number (remove + if present, ensure E.164 format)
    to_phone_normalized = to_phone.lstrip('+')
    
    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Build message payload
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_phone_normalized,
        "type": "video",
        "video": {
            "id": media_id
        }
    }
    
    # Add caption if provided
    if caption:
        payload["video"]["caption"] = caption
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                message_id = data.get('messages', [{}])[0].get('id')
                if message_id:
                    return message_id, None
                else:
                    return None, f"No message_id in response: {data}"
            else:
                error_text = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', error_text)
                except:
                    error_msg = error_text
                return None, f"Send failed (HTTP {response.status_code}): {error_msg}"
    
    except httpx.TimeoutException:
        return None, "Send timeout"
    except Exception as e:
        return None, f"Send error: {str(e)}"

