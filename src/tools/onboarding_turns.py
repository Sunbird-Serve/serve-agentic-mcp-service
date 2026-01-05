"""
Onboarding turn handler producing unified MCP responses with multi-intent support.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .faq_answer import FAQRequest as FAQAnswerRequest, faq_answer as faq_answer_tool
from .models import Fact, FactType, MCPResponse, evaluate_state_progress, normalize_fact_value
from .onboarding_parse import ParseRequest, ParseResponse, onboarding_parse_message

router = APIRouter()


class TurnRequest(BaseModel):
    """Input payload for onboarding.handle_turn."""

    state: str = Field(..., description="Current onboarding state, e.g., WELCOME")
    message: str = Field(..., description="Volunteer’s latest WhatsApp message")
    locale: str = Field(default="en-IN", description="Locale hint for parsing")
    known_facts: Optional[List[Fact]] = Field(
        default=None,
        description="Facts already captured in this session (server will merge with new extractions)",
    )
    policy_version: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None


def _fact_key(fact: Fact) -> Tuple[str, str]:
    value = fact.value
    if isinstance(value, (dict, list)):
        value_repr = json.dumps(value, sort_keys=True, default=str)
    else:
        value_repr = str(value)
    return (fact.type.value, value_repr)


def _merge_facts(existing: Optional[Sequence[Fact]], new: Sequence[Fact]) -> List[Fact]:
    merged: Dict[Tuple[str, str], Fact] = {}
    ordered: List[Fact] = []

    def _upsert(f: Fact) -> None:
        key = _fact_key(f)
        current = merged.get(key)
        if current is None or (f.confidence or 0.0) >= (current.confidence or 0.0):
            merged[key] = f

    for fact in existing or []:
        _upsert(fact)
    for fact in new:
        _upsert(fact)

    seen_keys: set[Tuple[str, str]] = set()
    for collection in ((existing or []), new):
        for fact in collection:
            key = _fact_key(fact)
            if key in merged and key not in seen_keys:
                ordered.append(merged[key])
                seen_keys.add(key)

    return ordered


def _extract_facts(parse: ParseResponse) -> List[Fact]:
    facts: List[Fact] = []

    if parse.consent and parse.consent.value != "unknown":
        facts.append(
            Fact(
                type=FactType.CONSENT,
                value=parse.consent.value,
                confidence=parse.consent.confidence,
                source="parser",
            )
        )

    if parse.eligibility:
        elig = parse.eligibility
        if elig.age_ok is not None or elig.age_years is not None:
            facts.append(
                Fact(
                    type=FactType.AGE,
                    value={
                        "years": elig.age_years,
                        "is_18_plus": elig.age_ok,
                    },
                    confidence=elig.confidence,
                    source="parser",
                )
            )
        if elig.device_ok is not None or elig.has_device is not None or elig.device_type:
            facts.append(
                Fact(
                    type=FactType.DEVICE,
                    value={
                        "has_device": elig.has_device,
                        "device_type": elig.device_type,
                        "device_ok": elig.device_ok,
                    },
                    confidence=elig.confidence,
                    source="parser",
                )
            )

    if parse.eligibility and parse.eligibility.weekly_commitment_hours is not None:
        facts.append(
            Fact(
                type=FactType.COMMITMENT_HOURS,
                value=parse.eligibility.weekly_commitment_hours,
                confidence=parse.eligibility.confidence,
                source="parser",
            )
        )

    if parse.availability:
        for item in parse.availability:
            if item.day:
                facts.append(
                    Fact(
                        type=FactType.PREFERRED_DAY,
                        value=item.day,
                        confidence=item.confidence,
                        source="parser",
                    )
                )
            if item.start and item.end:
                facts.append(
                    Fact(
                        type=FactType.PREFERRED_TIME,
                        value={"start": item.start, "end": item.end},
                        confidence=item.confidence,
                        source="parser",
                    )
                )

    if parse.days:
        conf = parse.prefs_confidence or 0.7
        for day in parse.days:
            facts.append(
                Fact(
                    type=FactType.PREFERRED_DAY,
                    value=day,
                    confidence=conf,
                    source="parser",
                )
            )

    if parse.time_windows:
        conf = parse.prefs_confidence or 0.7
        for window in parse.time_windows:
            facts.append(
                Fact(
                    type=FactType.PREFERRED_TIME,
                    value={"start": window.start, "end": window.end},
                    confidence=conf,
                    source="parser",
                )
            )

    return facts


def _state_label(state: str) -> str:
    mapping = {
        "WELCOME": "getting started",
        "ELIGIBILITY_PART1": "age and device basics",
        "ELIGIBILITY_PART2": "weekly commitment",
        "PREFS_DAYTIME": "weekday and time preferences",
        "QA_WINDOW": "any remaining questions",
        "ORIENTATION": "orientation scheduling",
    }
    return mapping.get(state.upper(), state.replace("_", " ").title())


def _format_ack(facts: Iterable[Fact]) -> List[str]:
    acknowledgements: List[str] = []
    days: List[str] = []
    time_windows: List[str] = []

    for fact in facts:
        if (fact.confidence or 0.0) < 0.6:
            continue

        if fact.type == FactType.CONSENT:
            val = str(fact.value).lower()
            if val == "yes":
                acknowledgements.append("Thanks for confirming you're ready to begin.")
            elif val == "no":
                acknowledgements.append("Understood, I'll pause the onboarding flow for now.")
            elif val == "defer":
                acknowledgements.append("Sure, we can revisit whenever you're ready.")
        elif fact.type == FactType.AGE:
            data = normalize_fact_value(FactType.AGE, fact.value) or {}
            years = data.get("years")
            if years:
                acknowledgements.append(f"Noted that you're {years}.")
            elif data.get("is_18_plus"):
                acknowledgements.append("Great, you're 18 or older.")
        elif fact.type == FactType.DEVICE:
            data = normalize_fact_value(FactType.DEVICE, fact.value) or {}
            if data.get("has_device"):
                dtype = data.get("device_type")
                if dtype:
                    acknowledgements.append(f"Good to know you have a {dtype}.")
                else:
                    acknowledgements.append("Glad you have a compatible device.")
        elif fact.type == FactType.COMMITMENT_HOURS:
            hours = normalize_fact_value(FactType.COMMITMENT_HOURS, fact.value)
            if hours is not None:
                acknowledgements.append(f"Thanks for sharing that you can offer about {hours:.1f} hours weekly.")
        elif fact.type == FactType.PREFERRED_DAY:
            day = normalize_fact_value(FactType.PREFERRED_DAY, fact.value)
            if isinstance(day, str):
                day_caps = day.title()
                if day_caps not in days:
                    days.append(day_caps)
        elif fact.type == FactType.PREFERRED_TIME:
            window = normalize_fact_value(FactType.PREFERRED_TIME, fact.value)
            if isinstance(window, dict):
                start = window.get("start")
                end = window.get("end")
                if start and end:
                    label = f"{start}-{end}"
                    if label not in time_windows:
                        time_windows.append(label)

    if days:
        if len(days) == 1:
            acknowledgements.append(f"Noted that {days[0]} works for you.")
        else:
            day_list = ", ".join(days[:-1]) + f" and {days[-1]}"
            acknowledgements.append(f"Noted those weekdays—{day_list}.")

    if time_windows:
        if len(time_windows) == 1:
            acknowledgements.append(f"I've captured the {time_windows[0]} window.")
        else:
            time_list = ", ".join(time_windows[:-1]) + f" and {time_windows[-1]}"
            acknowledgements.append(f"I've captured those windows: {time_list}.")

    return acknowledgements


def _compose_reply(
    state: str,
    acknowledgements: List[str],
    pending_questions: List[str],
    target_state: str,
) -> str:
    parts: List[str] = acknowledgements.copy()

    if pending_questions:
        parts.append(pending_questions[0])
    else:
        if target_state != "SAME":
            label = _state_label(target_state)
            parts.append(f"Next, let's cover {label}.")

    # ensure message is non-empty
    if not parts:
        parts.append("Thanks! Let's keep going.")

    return " ".join(part.strip() for part in parts if part.strip())


def _auto_advance(state: str, facts: Sequence[Fact]) -> Tuple[str, List[str]]:
    """
    Attempt to advance through multiple states using available facts.
    Returns the target next_state and pending questions for that state.
    """

    current = state.upper()
    pending_questions: List[str] = []
    visited: set[str] = set()

    while True:
        if current in visited:
            break
        visited.add(current)
        evaluation = evaluate_state_progress(current, facts)
        if evaluation.missing:
            # If we are still in the original state, signal SAME.
            next_state = "SAME" if current == state.upper() else current
            pending_questions = evaluation.pending_questions
            return next_state, pending_questions

        proposed = evaluation.next_state
        if proposed in ("SAME", current):
            next_state = "SAME" if current == state.upper() else current
            pending_questions = evaluation.pending_questions
            return next_state, pending_questions

        current = proposed

    return current, pending_questions


@router.post("/onboarding.handle_turn", response_model=MCPResponse)
async def onboarding_handle_turn(body: TurnRequest) -> MCPResponse:
    """
    Entry point for the onboarding agent. Parses the message, extracts facts,
    evaluates state progression, and returns a unified MCP response payload.
    """

    parse_request = ParseRequest(text=body.message, locale=body.locale, state=body.state)
    parse_result = await onboarding_parse_message(parse_request)

    new_facts = _extract_facts(parse_result)

    faq_answer_text: Optional[str] = None
    faq_debug: Optional[Dict[str, Any]] = None
    state_upper = body.state.upper()
    lower_message = body.message.lower()

    if state_upper == "QA_WINDOW":
        faq_request = FAQAnswerRequest(
            question=body.message,
            policy_context="SERVE onboarding policy guidance",
            state=body.state,
        )
        faq_response = await faq_answer_tool(faq_request)
        faq_debug = faq_response.model_dump()
        if faq_response.answer:
            faq_answer_text = faq_response.answer.strip()
            new_facts.append(
                Fact(
                    type=FactType.FAQ_ANSWER,
                    value=faq_answer_text,
                    confidence=faq_response.confidence,
                    source="faq.answer",
                )
            )

    if "orientation" in lower_message and (parse_result.availability or parse_result.days or parse_result.time_windows):
        new_facts.append(
            Fact(
                type=FactType.ORIENTATION_READY,
                value=True,
                confidence=0.75,
                source="parser",
            )
        )

    merged_facts = _merge_facts(body.known_facts, new_facts)

    next_state, pending_questions = _auto_advance(body.state, merged_facts)

    acknowledgements = _format_ack(new_facts)
    reply_text = _compose_reply(body.state, acknowledgements, pending_questions, next_state)
    if faq_answer_text:
        reply_text = f"{faq_answer_text} {reply_text}".strip() if reply_text else faq_answer_text

    debug_payload = {
        "state": body.state,
        "parse": parse_result.model_dump(),
        "new_facts": [fact.model_dump() for fact in new_facts],
        "known_facts": [fact.model_dump() for fact in body.known_facts or []],
    }
    if faq_debug:
        debug_payload["faq"] = faq_debug

    response = MCPResponse(
        reply=reply_text,
        next_state=next_state,
        facts=merged_facts,
        pending_questions=pending_questions or None,
        actions=None,
        debug=debug_payload,
    )

    return response.ensure_defaults()

