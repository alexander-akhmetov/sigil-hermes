"""In-flight recorder state.

Generations are keyed by ``(task_id, session_id, api_call_count)`` so we can
correlate the matching pre/post hook pair.

Generation state carries the parsed input messages alongside the recorder —
hermes's ``post_api_request`` hook does not reliably pass ``messages`` again
(matching the bundled langfuse plugin), so we capture them once at pre-time
and pass them back to ``set_result`` at post-time. Otherwise the SDK's
``set_result(input=[], output=...)`` would clear the input we seeded.

Tool executions don't need cross-hook state — we only register
``post_tool_call``, do all the work there, and close immediately.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class GenState:
    recorder: Any
    input_messages: list = field(default_factory=list)
    system_prompt: str = ""
    # Partial fields filled in by post_api_request — actual ``set_result`` is
    # deferred until post_llm_call so we can pair each call with its assistant
    # message from the final ``conversation_history``.
    usage: Any = None
    finish_reason: str = ""
    response_model: str = ""
    # Captured at pre_api_request and post_api_request so the generation span
    # and gen_ai.client.operation.duration metric reflect the LLM call alone,
    # not the close-deferred recorder lifetime that runs through tool execution
    # and any subsequent calls in the same turn.
    started_at: datetime | None = None
    api_duration: float | None = None


_GEN_STATE: dict[tuple[str, str, int], GenState] = {}
# Per-(task_id, session_id) running hermes-shaped message list. Populated by
# ``pre_llm_call`` from ``conversation_history`` and extended in-place as
# ``post_tool_call`` fires — so each ``pre_api_request`` snapshot reflects
# the messages going into THIS request, not the start-of-turn snapshot.
_CONVO_STATE: dict[tuple[str, str], list[dict]] = {}
# Count of ``role="assistant"`` messages present in ``conversation_history``
# at ``pre_llm_call`` time. Used by ``_close_pending_for_session`` to slice
# off prior turns' assistants and pair only this turn's outputs to recorders.
# A live count from ``_CONVO_STATE`` is wrong — ``post_tool_call`` extends
# the running convo with synthesized assistant tool-call messages, which we
# don't want included.
_TURN_START_ASST_COUNT: dict[tuple[str, str], int] = {}
_LOCK = threading.Lock()


def gen_put(key: tuple[str, str, int], state: GenState) -> None:
    with _LOCK:
        _GEN_STATE[key] = state


def gen_get(key: tuple[str, str, int]) -> GenState | None:
    with _LOCK:
        return _GEN_STATE.get(key)


def gen_pop(key: tuple[str, str, int]) -> GenState | None:
    with _LOCK:
        return _GEN_STATE.pop(key, None)


def gen_pop_session(session_id: str) -> list[tuple[tuple[str, str, int], GenState]]:
    """Pop and return all GenStates for a session, sorted by api_call_count."""
    with _LOCK:
        matching = [(k, v) for k, v in _GEN_STATE.items() if k[1] == session_id]
        for k, _ in matching:
            del _GEN_STATE[k]
    matching.sort(key=lambda kv: kv[0][2])
    return matching


def convo_set(key: tuple[str, str], messages: list[dict]) -> None:
    with _LOCK:
        _CONVO_STATE[key] = list(messages)


def convo_get(key: tuple[str, str]) -> list[dict]:
    with _LOCK:
        return list(_CONVO_STATE.get(key) or [])


def convo_append(key: tuple[str, str], message: dict) -> None:
    with _LOCK:
        if key in _CONVO_STATE:
            _CONVO_STATE[key].append(message)


def convo_clear(key: tuple[str, str]) -> None:
    with _LOCK:
        _CONVO_STATE.pop(key, None)


def turn_start_asst_count_set(key: tuple[str, str], count: int) -> None:
    with _LOCK:
        _TURN_START_ASST_COUNT[key] = count


def turn_start_asst_count_get(key: tuple[str, str]) -> int | None:
    with _LOCK:
        return _TURN_START_ASST_COUNT.get(key)


def turn_start_asst_count_clear(key: tuple[str, str]) -> None:
    with _LOCK:
        _TURN_START_ASST_COUNT.pop(key, None)


def reset_for_tests() -> None:
    with _LOCK:
        _GEN_STATE.clear()
        _CONVO_STATE.clear()
        _TURN_START_ASST_COUNT.clear()
