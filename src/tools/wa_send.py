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

class ListRow(BaseModel):
    """WhatsApp interactive list row definition"""
    id: str = Field(..., description="Unique row ID (will be returned when user selects this row)")
    title: str = Field(..., description="Row title (required, max 24 chars)")
    description: Optional[str] = Field(None, description="Row description (optional, max 72 chars)")

class ListSection(BaseModel):
    """WhatsApp interactive list section definition"""
    title: Optional[str] = Field(None, description="Section title (optional, max 24 chars)")
    rows: List[ListRow] = Field(..., description="List of rows in this section (required, 1-10 rows per section)")

class InteractiveList(BaseModel):
    """WhatsApp interactive list message definition"""
    body: str = Field(..., description="Message body text (required)")
    header: Optional[str] = Field(None, description="List header text (optional, max 60 chars)")
    footer: Optional[str] = Field(None, description="List footer text (optional, max 60 chars)")
    button: str = Field(..., description="Action button text (required, max 20 chars)")
    sections: List[ListSection] = Field(..., description="List of sections (required, 1-10 sections, each with 1-10 rows)")

class TemplateLanguage(BaseModel):
    """WhatsApp template language definition"""
    code: str = Field(..., description="Language code (e.g., 'en', 'hi')")

class Template(BaseModel):
    """WhatsApp template message definition"""
    name: str = Field(..., description="Template name (e.g., 'serve_welcome')")
    language: TemplateLanguage = Field(..., description="Template language")

class SendMessageInput(BaseModel):
    to: str = Field(..., description="E.164 number without +, e.g., 9198xxxxxxx")
    template: Optional[Template] = Field(
        None,
        description="Optional WhatsApp template message. If provided, this takes precedence over text/list/buttons"
    )
    text: Optional[str] = Field(None, description="Message text content (required if template/list is not provided)")
    buttons: Optional[List[Union[str, Button]]] = Field(
        None,
        description="Optional list of buttons. Can be list of strings (quick replies) or list of Button objects with id/title (used if template/list is not provided)"
    )
    interactive_list: Optional[InteractiveList] = Field(
        None,
        alias="list",
        description="Optional interactive list message. If provided, this will be used instead of text/buttons (template takes precedence). JSON field name is 'list'"
    )

async def publish_wa_out(
    to: str, 
    text: Optional[str] = None, 
    buttons: Optional[List[Union[str, Button]]] = None,
    interactive_list: Optional[InteractiveList] = None,
    template: Optional[Template] = None
):
    """
    Publish WhatsApp outbound message to Kafka.
    
    Priority order: template > interactive_list > text/buttons
    
    Args:
        to: Phone number in E.164 format without +
        text: Message text content (used if template/interactive_list is not provided)
        buttons: Optional list of buttons (strings or Button objects) (used if template/interactive_list is not provided)
        interactive_list: Optional interactive list message (used if template is not provided)
        template: Optional template message (takes highest precedence)
    """
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    await producer.start()
    try:
        # Build event data
        event_data = {"to": to}
        
        # Priority: template > interactive_list > text/buttons
        if template:
            # Template message
            event_data["template"] = {
                "name": template.name,
                "language": {
                    "code": template.language.code
                }
            }
        elif interactive_list:
            # Convert InteractiveList model to dict
            list_dict = {
                "body": _normalize_wa_text(interactive_list.body),
                "button": interactive_list.button,
                "sections": [
                    {
                        "rows": [
                            {
                                "id": row.id,
                                "title": row.title,
                                **({"description": row.description} if row.description else {})
                            }
                            for row in section.rows
                        ],
                        **({"title": section.title} if section.title else {})
                    }
                    for section in interactive_list.sections
                ]
            }
            if interactive_list.header:
                list_dict["header"] = _normalize_wa_text(interactive_list.header)
            if interactive_list.footer:
                list_dict["footer"] = _normalize_wa_text(interactive_list.footer)
            
            event_data["list"] = list_dict
        else:
            # Regular text message with optional buttons
            safe_text = _normalize_wa_text(text)
            event_data["text"] = safe_text
            
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
