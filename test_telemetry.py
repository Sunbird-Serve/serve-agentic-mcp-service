"""
Unit tests for Telemetry MCP tool.
Tests the telemetry router independently without needing the full app.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ["TELEMETRY_LOG_DIR"] = "/tmp/telemetry_test_logs"

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    """Create a minimal FastAPI app with just the telemetry router."""
    from tools.telemetry import router, _events, _lock
    with _lock:
        _events.clear()
    test_app = FastAPI()
    test_app.include_router(router, prefix="/mcp")
    return test_app


@pytest.fixture(autouse=True)
def clear_events():
    """Clear in-memory events before each test."""
    from tools.telemetry import _events, _lock
    with _lock:
        _events.clear()
    yield
    with _lock:
        _events.clear()


@pytest.mark.asyncio
async def test_emit_basic(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.completed",
            "payload": {"step": "done"},
            "volunteerId": "vol-123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["event"] == "onboarding.completed"
        assert data["eventId"].startswith("evt_")
        assert data["timestamp"].endswith("Z")


@pytest.mark.asyncio
async def test_emit_unknown_event_type(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/telemetry.emit", json={
            "event": "custom.unknown_event",
            "payload": {"info": "test"}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True


@pytest.mark.asyncio
async def test_emit_minimal_payload(app):
    """Event with only required field (event type)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/telemetry.emit", json={
            "event": "tool.called",
            "payload": {}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True


@pytest.mark.asyncio
async def test_query_by_volunteer(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.started", "payload": {}, "volunteerId": "vol-A"
        })
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.started", "payload": {}, "volunteerId": "vol-B"
        })
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.completed", "payload": {}, "volunteerId": "vol-A"
        })

        resp = await client.post("/mcp/telemetry.query", json={
            "volunteerId": "vol-A"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(e["volunteerId"] == "vol-A" for e in data["events"])


@pytest.mark.asyncio
async def test_query_by_event_type_prefix(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.started", "payload": {}
        })
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.completed", "payload": {}
        })
        await client.post("/mcp/telemetry.emit", json={
            "event": "tool.called", "payload": {}
        })

        resp = await client.post("/mcp/telemetry.query", json={
            "eventType": "onboarding"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2


@pytest.mark.asyncio
async def test_query_limit(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(10):
            await client.post("/mcp/telemetry.emit", json={
                "event": "tool.called", "payload": {"i": i}
            })

        resp = await client.post("/mcp/telemetry.query", json={
            "limit": 3
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) == 3
        assert data["total"] == 10
        assert data["truncated"] is True


@pytest.mark.asyncio
async def test_stats(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.started", "payload": {}, "volunteerId": "v1"
        })
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.started", "payload": {}, "volunteerId": "v2"
        })
        await client.post("/mcp/telemetry.emit", json={
            "event": "tool.called", "payload": {}, "volunteerId": "v1"
        })

        resp = await client.post("/mcp/telemetry.stats", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 3
        assert data["by_type"]["onboarding.started"] == 2
        assert data["by_type"]["tool.called"] == 1
        assert data["unique_volunteers"] == 2


@pytest.mark.asyncio
async def test_stats_filtered_by_volunteer(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.started", "payload": {}, "volunteerId": "v1"
        })
        await client.post("/mcp/telemetry.emit", json={
            "event": "onboarding.completed", "payload": {}, "volunteerId": "v2"
        })

        resp = await client.post("/mcp/telemetry.stats", json={
            "volunteerId": "v1"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 1
        assert data["unique_volunteers"] == 1
