"""
Unit tests for State MCP: Conversation State Store.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ["STATE_STORE_DIR"] = "/tmp/state_store_test"
os.environ["STATE_TTL_HOURS"] = "72"

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from tools.state import router, _STATE_STORE, _TRANSITION_STORE, _CONVERSATION_STORE, _lock
    with _lock:
        _STATE_STORE.clear()
        _TRANSITION_STORE.clear()
        _CONVERSATION_STORE.clear()
    test_app = FastAPI()
    test_app.include_router(router, prefix="/mcp")
    return test_app


@pytest.fixture(autouse=True)
def clear_stores():
    from tools.state import _STATE_STORE, _TRANSITION_STORE, _CONVERSATION_STORE, _lock
    with _lock:
        _STATE_STORE.clear()
        _TRANSITION_STORE.clear()
        _CONVERSATION_STORE.clear()
    yield


@pytest.mark.asyncio
async def test_get_default_state(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/state.get", json={"volunteerId": "new-vol"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "WELCOME"
        assert data["expired"] is False


@pytest.mark.asyncio
async def test_advance_state(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/state.advance", json={
            "volunteerId": "vol-1",
            "intent": "to_ELIGIBILITY_PART1"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["new_state"] == "ELIGIBILITY_PART1"
        assert data["previous_state"] == "WELCOME"
        assert data["next_required_fields"] == ["age_ok", "has_device"]


@pytest.mark.asyncio
async def test_advance_invalid_transition(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First advance to ELIGIBILITY_PART1
        await client.post("/mcp/state.advance", json={
            "volunteerId": "vol-1", "intent": "to_ELIGIBILITY_PART1"
        })
        # Try to skip ahead to SCHEDULING (invalid)
        resp = await client.post("/mcp/state.advance", json={
            "volunteerId": "vol-1", "intent": "to_SCHEDULING"
        })
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_advance_idempotency(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        key = "idem-key-123"
        resp1 = await client.post("/mcp/state.advance", json={
            "volunteerId": "vol-1", "intent": "to_ELIGIBILITY_PART1", "idempotency_key": key
        })
        resp2 = await client.post("/mcp/state.advance", json={
            "volunteerId": "vol-1", "intent": "to_ELIGIBILITY_PART1", "idempotency_key": key
        })
        assert resp1.json()["transitioned_at"] == resp2.json()["transitioned_at"]


@pytest.mark.asyncio
async def test_terminal_state_from_any(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/state.advance", json={
            "volunteerId": "vol-1", "intent": "to_REJECTED"
        })
        assert resp.status_code == 200
        assert resp.json()["new_state"] == "REJECTED"


@pytest.mark.asyncio
async def test_conversation_save_and_get(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Save conversation state
        resp = await client.post("/mcp/conversation.save", json={
            "volunteerId": "vol-1",
            "state": {"step": "collecting_name", "facts": {"age": 25}, "messages": ["hi"]}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert data["version"] == 1
        assert "expires_at" in data

        # Get it back
        resp = await client.post("/mcp/conversation.get", json={"volunteerId": "vol-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["state"]["step"] == "collecting_name"
        assert data["state"]["facts"]["age"] == 25
        assert data["expired"] is False
        assert data["version"] == 1


@pytest.mark.asyncio
async def test_conversation_get_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/conversation.get", json={"volunteerId": "unknown"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False


@pytest.mark.asyncio
async def test_conversation_version_increments(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/mcp/conversation.save", json={
            "volunteerId": "vol-1", "state": {"v": 1}
        })
        resp = await client.post("/mcp/conversation.save", json={
            "volunteerId": "vol-1", "state": {"v": 2}
        })
        assert resp.json()["version"] == 2

        resp = await client.post("/mcp/conversation.get", json={"volunteerId": "vol-1"})
        assert resp.json()["state"]["v"] == 2
