import json
from pydantic import BaseModel, Field
from aiokafka import AIOKafkaProducer
from config import settings

class SendMessageInput(BaseModel):
    to: str = Field(..., description="E.164 number without +, e.g., 9198xxxxxxx")
    text: str

async def publish_wa_out(to: str, text: str):
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v).encode(),
        key_serializer=lambda k: k.encode() if k else None,
    )
    await producer.start()
    try:
        event = {"type": "wa.outbound.v1", "data": {"to": to, "text": text}}
        await producer.send_and_wait(settings.TOPIC_WA_OUT, key=to, value=event)
        return {"ok": True, "event": event}
    finally:
        await producer.stop()
