"""
Unit tests for Rule-Based Matching Engine.
Tests scoring logic independently without external API calls.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from tools.matching import (
    _normalize_day,
    _time_to_minutes,
    _time_overlap_minutes,
    _calculate_day_score,
    _calculate_time_score,
    _calculate_recency_score,
    _score_need,
)


class TestDayNormalization:
    def test_lowercase_short(self):
        assert _normalize_day("mon") == "MONDAY"
        assert _normalize_day("fri") == "FRIDAY"

    def test_full_name(self):
        assert _normalize_day("Monday") == "MONDAY"
        assert _normalize_day("WEDNESDAY") == "WEDNESDAY"

    def test_unknown_passes_through(self):
        assert _normalize_day("XYZ") == "XYZ"


class TestTimeToMinutes:
    def test_midnight(self):
        assert _time_to_minutes("00:00") == 0

    def test_noon(self):
        assert _time_to_minutes("12:00") == 720

    def test_afternoon(self):
        assert _time_to_minutes("14:30") == 870


class TestTimeOverlap:
    def test_full_overlap(self):
        assert _time_overlap_minutes(480, 660, 480, 660) == 180  # 8:00-11:00

    def test_partial_overlap(self):
        assert _time_overlap_minutes(480, 660, 600, 720) == 60  # overlap 10:00-11:00

    def test_no_overlap(self):
        assert _time_overlap_minutes(480, 600, 660, 720) == 0

    def test_contained(self):
        assert _time_overlap_minutes(480, 720, 540, 660) == 120  # 9:00-11:00 within 8:00-12:00


class TestDayScore:
    def test_perfect_match(self):
        score, matched = _calculate_day_score(["Mon", "Tue", "Wed"], ["MONDAY", "TUESDAY", "WEDNESDAY"])
        assert score == 1.0
        assert len(matched) == 3

    def test_partial_match(self):
        score, matched = _calculate_day_score(["Mon", "Tue"], ["MONDAY", "TUESDAY", "WEDNESDAY"])
        assert round(score, 2) == 0.67
        assert len(matched) == 2

    def test_no_match(self):
        score, matched = _calculate_day_score(["Sat", "Sun"], ["MONDAY", "TUESDAY"])
        assert score == 0.0
        assert len(matched) == 0

    def test_empty_need_days(self):
        score, matched = _calculate_day_score(["Mon", "Tue"], [])
        assert score == 1.0


class TestTimeScore:
    def test_full_time_overlap(self):
        vol_windows = [{"start": "08:00", "end": "11:00"}]
        need_slots = [{"startTime": "08:00", "endTime": "11:00", "day": "MONDAY"}]
        score, matched = _calculate_time_score(vol_windows, need_slots)
        assert score == 1.0
        assert len(matched) == 1

    def test_partial_time_overlap(self):
        vol_windows = [{"start": "09:00", "end": "11:00"}]
        need_slots = [{"startTime": "08:00", "endTime": "12:00", "day": "MONDAY"}]
        score, matched = _calculate_time_score(vol_windows, need_slots)
        assert 0.0 < score < 1.0
        assert len(matched) == 1

    def test_no_time_overlap(self):
        vol_windows = [{"start": "08:00", "end": "10:00"}]
        need_slots = [{"startTime": "14:00", "endTime": "16:00", "day": "MONDAY"}]
        score, matched = _calculate_time_score(vol_windows, need_slots)
        assert score == 0.0

    def test_empty_windows(self):
        score, matched = _calculate_time_score([], [{"startTime": "08:00", "endTime": "10:00"}])
        assert score == 0.5  # Default when no volunteer windows


class TestRecencyScore:
    def test_past_end_date(self):
        score = _calculate_recency_score("2020-01-01", "2020-12-31")
        assert score == 0.0

    def test_no_dates(self):
        score = _calculate_recency_score("", "")
        assert score == 0.5


class TestScoreNeed:
    def test_good_match(self):
        result = _score_need(
            vol_days=["Mon", "Tue", "Wed"],
            vol_windows=[{"start": "08:00", "end": "11:00"}],
            need_item={
                "needId": "need-1",
                "title": "Math Teaching",
                "schoolName": "ABC School",
                "days": ["MONDAY", "TUESDAY"],
                "timeSlots": [{"day": "MONDAY", "startTime": "09:00", "endTime": "10:00"}],
                "startDate": "2026-06-10",
                "endDate": "2026-12-31",
            }
        )
        assert result is not None
        assert result.score > 0.5
        assert len(result.matchedDays) >= 2
        assert result.needId == "need-1"

    def test_zero_day_overlap_still_has_score(self):
        result = _score_need(
            vol_days=["Sat", "Sun"],
            vol_windows=[{"start": "08:00", "end": "17:00"}],
            need_item={
                "needId": "need-2",
                "title": "Science Class",
                "schoolName": "XYZ School",
                "days": ["MONDAY", "TUESDAY"],
                "timeSlots": [{"day": "MONDAY", "startTime": "09:00", "endTime": "10:00"}],
                "startDate": "2026-06-10",
                "endDate": "2026-12-31",
            }
        )
        assert result is not None
        # Still has time+recency score even without day match
        assert result.score > 0.0
        assert len(result.matchedDays) == 0
