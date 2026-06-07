"""
Unit tests for Volunteer Profiles & Preferences MCP tools.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ["PROFILE_STORE_DIR"] = "/tmp/profile_store_test"

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from tools.profile import router, _PROFILE_STORE, _lock
    with _lock:
        _PROFILE_STORE.clear()
    test_app = FastAPI()
    test_app.include_router(router, prefix="/mcp")
    return test_app


@pytest.fixture(autouse=True)
def clear_store():
    from tools.profile import _PROFILE_STORE, _lock
    with _lock:
        _PROFILE_STORE.clear()
    yield


@pytest.mark.asyncio
async def test_save_profile(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/profile.save", json={
            "volunteerId": "vol-1",
            "profile": {
                "identity": {"fullname": "Priya Sharma"},
                "contact": {"email": "priya@example.com", "mobile": "+919876543210"},
                "teaching": {
                    "subjects": ["Mathematics", "Science"],
                    "grades": ["6", "7", "8"],
                    "languages": ["English", "Hindi"],
                    "days": ["Mon", "Wed", "Fri"],
                    "time_windows": [{"start": "09:00", "end": "11:00"}],
                    "timezone": "Asia/Kolkata"
                },
                "eligibility": {"age_years": 28, "has_device": True, "weekly_commitment_hours": 4},
                "consent_given": True
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert data["version"] == 1


@pytest.mark.asyncio
async def test_get_profile(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/mcp/profile.save", json={
            "volunteerId": "vol-1",
            "profile": {
                "identity": {"fullname": "Ravi Kumar"},
                "teaching": {"subjects": ["English"]}
            }
        })

        resp = await client.post("/mcp/profile.get", json={"volunteerId": "vol-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["profile"]["identity"]["fullname"] == "Ravi Kumar"
        assert data["profile"]["teaching"]["subjects"] == ["English"]
        assert data["version"] == 1


@pytest.mark.asyncio
async def test_get_profile_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/profile.get", json={"volunteerId": "unknown"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False


@pytest.mark.asyncio
async def test_update_profile_partial(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create initial profile
        await client.post("/mcp/profile.save", json={
            "volunteerId": "vol-1",
            "profile": {
                "identity": {"fullname": "Anita Desai"},
                "teaching": {"subjects": ["Math"], "grades": ["5"]}
            }
        })

        # Partial update — add a subject
        resp = await client.post("/mcp/profile.update", json={
            "volunteerId": "vol-1",
            "updates": {
                "teaching": {"subjects": ["Math", "Science"], "languages": ["English"]}
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"] is True
        assert data["version"] == 2
        assert "teaching" in data["changed_fields"]

        # Verify merged state
        resp = await client.post("/mcp/profile.get", json={"volunteerId": "vol-1"})
        profile = resp.json()["profile"]
        assert profile["identity"]["fullname"] == "Anita Desai"  # preserved
        assert profile["teaching"]["subjects"] == ["Math", "Science"]  # updated
        assert profile["teaching"]["languages"] == ["English"]  # added
        assert profile["teaching"]["grades"] == ["5"]  # preserved from deep merge


@pytest.mark.asyncio
async def test_update_creates_if_missing(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/profile.update", json={
            "volunteerId": "new-vol",
            "updates": {"identity": {"fullname": "New Person"}, "consent_given": True}
        })
        assert resp.status_code == 200
        assert resp.json()["version"] == 1

        resp = await client.post("/mcp/profile.get", json={"volunteerId": "new-vol"})
        assert resp.json()["found"] is True
        assert resp.json()["profile"]["identity"]["fullname"] == "New Person"


@pytest.mark.asyncio
async def test_version_increments(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/mcp/profile.save", json={
            "volunteerId": "vol-1",
            "profile": {"identity": {"fullname": "V1"}}
        })
        resp = await client.post("/mcp/profile.save", json={
            "volunteerId": "vol-1",
            "profile": {"identity": {"fullname": "V2"}}
        })
        assert resp.json()["version"] == 2
