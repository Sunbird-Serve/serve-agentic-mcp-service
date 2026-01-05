import json
from typing import Optional, List, Union
from pydantic import BaseModel, Field
from aiokafka import AIOKafkaProducer
from config import settings

def _normalize_wa_text(text: str) -> str:
    """Optionally strip non-ASCII for downstream if required; otherwise keep UTF-8."""
    if getattr(settings, "WA_ASCII_ONLY", False):
        return text.encode("ascii", errors="ignore").decode("ascii")
    return text

class Button(BaseModel):
    """WhatsApp button definition"""
    id: str = Field(..., description="Unique button ID")
    title: str = Field(..., description="Button label text")

class SendMessageInput(BaseModel):
    to: str = Field(..., description="E.164 number without +, e.g., 9198xxxxxxx")
    text: str
    buttons: Optional[List[Union[str, Button]]] = Field(
        None,
        description="Optional list of buttons. Can be list of strings (quick replies) or list of Button objects with id/title"
    )

async def publish_wa_out(to: str, text: str, buttons: Optional[List[Union[str, Button]]] = None):
    """
    Publish WhatsApp outbound message to Kafka.
    
    Args:
        to: Phone number in E.164 format without +
        text: Message text content
        buttons: Optional list of buttons (strings or Button objects)
    """
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    await producer.start()
    try:
        safe_text = _normalize_wa_text(text)
        
        # Build event data
        event_data = {"to": to, "text": safe_text}
        
        # Add buttons if provided
        if buttons:
            # Normalize buttons: convert strings to Button objects, or use Button objects as-is
            normalized_buttons = []
            for btn in buttons:
                if isinstance(btn, str):
                    # Simple string button - use as both id and title
                    normalized_buttons.append({"id": btn.lower().replace(" ", "_"), "title": btn})
                elif isinstance(btn, Button):
                    # Button object - convert to dict
                    normalized_buttons.append({"id": btn.id, "title": btn.title})
                elif isinstance(btn, dict):
                    # Already a dict (from JSON) - use as-is
                    normalized_buttons.append(btn)
            
            event_data["buttons"] = normalized_buttons
        
        event = {"type": "wa.outbound.v1", "data": event_data}
        await producer.send_and_wait(settings.TOPIC_WA_OUT, key=to, value=event)
        return {"ok": True, "event": event}
    finally:
        await producer.stop()
