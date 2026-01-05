#!/usr/bin/env python3
"""
Regression checks for onboarding.handle_turn multi-intent handling.

Run manually with: python test_onboarding_turns.py
"""
import asyncio
from typing import List

from src.tools.models import Fact, FactType
from src.tools.onboarding_turns import TurnRequest, onboarding_handle_turn


def _has_fact(facts: List[Fact], fact_type: FactType) -> bool:
    return any(f.type == fact_type for f in facts)


def _get_fact_values(facts: List[Fact], fact_type: FactType):
    return [f.value for f in facts if f.type == fact_type]


async def _test_welcome_multi_intent() -> None:
    req = TurnRequest(
        state="WELCOME",
        message="Yes, please start onboarding. I'm 23 and have a laptop ready.",
        locale="en-IN",
    )
    resp = await onboarding_handle_turn(req)
    assert resp.next_state == "ELIGIBILITY_PART2", resp.next_state
    assert _has_fact(resp.facts, FactType.CONSENT)
    assert _has_fact(resp.facts, FactType.AGE)
    assert _has_fact(resp.facts, FactType.DEVICE)


async def _test_eligibility_part2_carries_availability() -> None:
    req = TurnRequest(
        state="ELIGIBILITY_PART2",
        message="I can give 2 hours weekly, preferably Monday 9-10 AM and Wednesday after lunch.",
        locale="en-IN",
    )
    resp = await onboarding_handle_turn(req)
    assert resp.next_state == "QA_WINDOW", resp.next_state
    assert _has_fact(resp.facts, FactType.COMMITMENT_HOURS)
    assert _has_fact(resp.facts, FactType.PREFERRED_DAY)
    assert _has_fact(resp.facts, FactType.PREFERRED_TIME)


async def _test_qa_orientation_combo() -> None:
    req = TurnRequest(
        state="QA_WINDOW",
        message="Is this paid? If not, I can attend orientation next Tuesday at 10 am.",
        locale="en-IN",
    )
    resp = await onboarding_handle_turn(req)
    assert resp.next_state == "ORIENTATION", resp.next_state
    assert "100% volunteer" in resp.reply
    assert _has_fact(resp.facts, FactType.FAQ_ANSWER)
    assert _has_fact(resp.facts, FactType.ORIENTATION_READY)
    # Orientation intent should still carry the availability clues forward
    day_values = _get_fact_values(resp.facts, FactType.PREFERRED_DAY)
    time_values = _get_fact_values(resp.facts, FactType.PREFERRED_TIME)
    assert day_values, "Expected weekday preference facts"
    assert time_values, "Expected time window facts"


async def main() -> None:
    await _test_welcome_multi_intent()
    await _test_eligibility_part2_carries_availability()
    await _test_qa_orientation_combo()
    print("✅ onboarding.handle_turn regression checks passed.")


if __name__ == "__main__":
    asyncio.run(main())

