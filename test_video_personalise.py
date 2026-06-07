"""
Unit tests for Personalised Video MCP tool.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from tools.video_personalise import router
    test_app = FastAPI()
    test_app.include_router(router, prefix="/mcp")
    return test_app


@pytest.mark.asyncio
async def test_generate_welcome_video(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/video.generate_personalised", json={
            "name": "Priya",
            "context": "welcome"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Priya" in data["caption"]
        assert data["videoType"] == "welcome"
        assert data["personalisedFor"] == "Priya"


@pytest.mark.asyncio
async def test_generate_orientation_video(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/video.generate_personalised", json={
            "name": "Ravi Kumar",
            "context": "orientation"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Ravi Kumar" in data["caption"]
        assert data["videoType"] == "orientation"


@pytest.mark.asyncio
async def test_generate_thankyou_video(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/video.generate_personalised", json={
            "name": "Anita",
            "context": "thankyou"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Anita" in data["caption"]
        assert "Thank you" in data["caption"]


@pytest.mark.asyncio
async def test_generate_with_custom_message(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/video.generate_personalised", json={
            "name": "Priya",
            "context": "welcome",
            "custom_message": "Your orientation is on Monday at 10 AM."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Priya" in data["caption"]
        assert "Monday at 10 AM" in data["caption"]


@pytest.mark.asyncio
async def test_generate_invalid_context(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/video.generate_personalised", json={
            "name": "Priya",
            "context": "invalid_context"
        })
        assert resp.status_code == 422  # Pydantic validation error (enum)


@pytest.mark.asyncio
async def test_send_personalised_no_whatsapp_config(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/video.send_personalised", json={
            "to_phone": "+919876543210",
            "name": "Priya",
            "context": "welcome"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "not configured" in data["error"]
