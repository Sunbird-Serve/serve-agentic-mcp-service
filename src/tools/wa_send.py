import json
from pydantic import BaseModel, Field
from aiokafka import AIOKafkaProducer
from config import settings

def _normalize_wa_text(text: str) -> str:
    """Optionally strip non-ASCII for downstream if required; otherwise keep UTF-8."""
    if getattr(settings, "WA_ASCII_ONLY", False):
        return text.encode("ascii", errors="ignore").decode("ascii")
    return text

class SendMessageInput(BaseModel):
    to: str = Field(..., description="E.164 number without +, e.g., 9198xxxxxxx")
    text: str

async def publish_wa_out(to: str, text: str):
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    await producer.start()
    try:
        safe_text = _normalize_wa_text(text)
        event = {"type": "wa.outbound.v1", "data": {"to": to, "text": safe_text}}
        await producer.send_and_wait(settings.TOPIC_WA_OUT, key=to, value=event)
        return {"ok": True, "event": event}
    finally:
        await producer.stop()
