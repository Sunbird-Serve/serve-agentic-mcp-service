"""
Unit tests for Serve Needs (getNeedDetails) and Volunteer Status tools.
Tests response models and helpers without calling external APIs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from tools.serve_needs import _map_serve_response_item, _parse_iso_date, _parse_time_to_hhmm
from tools.serve_status import _extract_onboard_status, _extract_nominations


class TestNeedsHelpers:
    def test_parse_iso_date(self):
        assert _parse_iso_date("2026-06-10T00:00:00Z") == "2026-06-10"
        assert _parse_iso_date("2026-06-10") == "2026-06-10"
        assert _parse_iso_date("") == ""
        assert _parse_iso_date(None) == ""

    def test_parse_time_to_hhmm(self):
        assert _parse_time_to_hhmm("2026-06-10T09:30:00+05:30") == "09:30"
        assert _parse_time_to_hhmm("14:00:00") == "14:00"
        assert _parse_time_to_hhmm("") == ""

    def test_map_serve_response_item(self):
        item = {
            "id": "need-uuid-123",
            "need": {"name": "Math Teaching", "needPurpose": "Education"},
            "entity": {"name": "ABC School", "district": "Chennai", "state": "Tamil Nadu"},
            "occurrence": {"startDate": "2026-06-10T00:00:00Z", "endDate": "2026-12-31T00:00:00Z"},
            "status": "Approved",
            "timeSlots": [
                {"day": "Monday", "startTime": "2026-06-10T09:00:00+05:30", "endTime": "2026-06-10T10:00:00+05:30"},
                {"day": "Wednesday", "startTime": "2026-06-10T09:00:00+05:30", "endTime": "2026-06-10T10:00:00+05:30"}
            ]
        }
        result = _map_serve_response_item(item)
        assert result.needId == "need-uuid-123"
        assert result.title == "Math Teaching"
        assert result.schoolName == "ABC School"
        assert result.district == "Chennai"
        assert "MONDAY" in result.days
        assert "WEDNESDAY" in result.days
        assert len(result.timeSlots) == 2
        assert result.timeSlots[0].startTime == "09:00"


class TestStatusHelpers:
    def test_extract_onboard_status(self):
        profile_data = {
            "onboardDetails": {
                "onboardStatus": [
                    {"onboardStep": "Discussion", "status": "completed"},
                    {"onboardStep": "Training", "status": "in-progress"}
                ],
                "profileCompletion": "75"
            }
        }
        assert _extract_onboard_status(profile_data) == "Training:in-progress"

    def test_extract_onboard_status_empty(self):
        assert _extract_onboard_status({}) == ""
        assert _extract_onboard_status({"onboardDetails": {}}) == ""

    def test_extract_nominations(self):
        fulfillments = [
            {
                "needId": "need-1",
                "need": {"id": "need-1", "name": "Math Class"},
                "status": "Nominated",
                "createdAt": "2026-06-01T10:00:00Z"
            },
            {
                "needId": "need-2",
                "need": {"id": "need-2", "name": "Science Class"},
                "status": "Confirmed",
                "createdAt": "2026-06-05T10:00:00Z"
            }
        ]
        result = _extract_nominations(fulfillments)
        assert len(result) == 2
        assert result[0].needId == "need-1"
        assert result[0].needTitle == "Math Class"
        assert result[1].status == "Confirmed"

    def test_extract_nominations_empty(self):
        assert _extract_nominations([]) == []
        assert _extract_nominations([None, "invalid"]) == []
