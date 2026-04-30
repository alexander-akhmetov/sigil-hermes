"""Hermes plugin hook handlers.

All handlers fail open: any exception is caught, logged at most once per kind,
and the hermes loop is allowed to continue. If the Sigil client cannot be
constructed (missing creds, SDK error), every handler short-circuits via
``client is None``.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from . import _client, _redact, _state

logger = logging.getLogger(__name__)


def _agent_name() -> str:
    cfg = _client._get_plugin_config()
    return cfg.agent_name if cfg is not None else "hermes"


def _convo_key(task_id: str, session_id: str) -> tuple[str, str]:
    """Key for the running conversation history.

    Hermes does not pass ``task_id`` to ``pre_llm_call`` but does to
    ``pre_api_request`` — keying on session_id only is the only way both hooks
    address the same bucket. Wrapped in a tuple so the type matches what
    ``_state`` expects.
    """
    return ("", session_id or "")


def _should_sample() -> bool:
    """Return True if this trace should be recorded under HERMES_SIGIL_SAMPLE_RATE.

    A pre-hook that returns False simply skips ``start_generation`` and
    never stores a recorder, so the matching post-hook becomes a natural
    no-op (``gen_pop`` returns None). Tool sampling is checked at
    ``post_tool_call`` time directly.
    """
    cfg = _client._get_plugin_config()
    if cfg is None or cfg.sample_rate >= 1.0:
        return True
    if cfg.sample_rate <= 0.0:
        return False
    return random.random() < cfg.sample_rate


def _coerce_text(content: Any) -> str:
    """Best-effort conversion of a hermes message ``content`` field to a string.

    Hermes mirrors OpenAI/Anthropic shapes — content can be a string or a list
    of typed blocks (``{"type": "text", "text": "..."}``). We collapse list
    blocks into newline-joined text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    chunks.append(block["text"])
                elif isinstance(block.get("content"), str):
                    chunks.append(block["content"])
                else:
                    chunks.append(json.dumps(block, default=str))
            else:
                chunks.append(repr(block))
        return "\n".join(c for c in chunks if c)
    return str(content)


def _split_system_prompt(messages: Any) -> tuple[str, list[dict]]:
    """Pull system messages out into a single prompt string, return remaining messages."""
    if not isinstance(messages, list):
        return "", []
    system_parts: list[str] = []
    rest: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            text = _coerce_text(msg.get("content"))
            if text:
                system_parts.append(text)
        else:
            rest.append(msg)
    return "\n\n".join(system_parts), rest


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            arguments = fn.get("arguments") if isinstance(fn, dict) else None
        else:
            tc_id = getattr(tc, "id", "")
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            arguments = getattr(fn, "arguments", None) if fn is not None else None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass
        out.append({"id": tc_id or "", "name": name or "", "arguments": arguments})
    return out


def _hermes_message_to_sigil(msg: dict[str, Any]):
    """Convert one hermes-shaped message dict to a sigil_sdk.Message."""
    from sigil_sdk import (
        Message,
        MessageRole,
        Part,
        ToolCall,
        ToolResult,
        text_part,
        tool_call_part,
        tool_result_part,
    )

    role = msg.get("role")
    if role == "user":
        text = _coerce_text(msg.get("content"))
        return Message(role=MessageRole.USER, parts=[text_part(text)] if text else [])
    if role == "tool":
        tool_call_id = msg.get("tool_call_id") or ""
        content = _coerce_text(msg.get("content"))
        return Message(
            role=MessageRole.TOOL,
            parts=[tool_result_part(ToolResult(tool_call_id=tool_call_id, content=content))],
        )
    if role == "assistant":
        parts: list[Part] = []
        text = _coerce_text(msg.get("content"))
        if text:
            parts.append(text_part(text))
        for tc in _serialize_tool_calls(msg.get("tool_calls")):
            input_json = b""
            if tc.get("arguments") is not None:
                try:
                    input_json = json.dumps(tc["arguments"]).encode()
                except Exception:
                    input_json = b""
            parts.append(
                tool_call_part(
                    ToolCall(name=tc.get("name", ""), id=tc.get("id", ""), input_json=input_json)
                )
            )
        return Message(role=MessageRole.ASSISTANT, parts=parts)
    # Unknown role (e.g. "system" should already be filtered out): drop.
    return None


def _messages_to_sigil(messages: Any) -> list:
    if not isinstance(messages, list):
        return []
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            continue  # handled via system_prompt
        sigil_msg = _hermes_message_to_sigil(msg)
        if sigil_msg is not None:
            out.append(sigil_msg)
    return out


def _assistant_message_to_sigil(assistant_message: Any) -> list:
    """Build a single-element list[Message] from a hermes assistant response."""
    if assistant_message is None:
        return []

    if isinstance(assistant_message, dict):
        content = assistant_message.get("content")
        tool_calls = assistant_message.get("tool_calls")
    else:
        content = getattr(assistant_message, "content", None)
        tool_calls = getattr(assistant_message, "tool_calls", None)

    msg_dict = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    sigil = _hermes_message_to_sigil(msg_dict)
    return [sigil] if sigil is not None else []


def _build_token_usage(usage: Any):
    from sigil_sdk import TokenUsage

    if not isinstance(usage, dict):
        return TokenUsage()

    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    cache_read = int(
        usage.get("cache_read_tokens")
        or usage.get("cache_read_input_tokens")
        or 0
    )
    cache_write = int(
        usage.get("cache_write_tokens")
        or usage.get("cache_creation_input_tokens")
        or usage.get("cache_write_input_tokens")
        or 0
    )
    reasoning = int(usage.get("reasoning_tokens") or 0)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning,
    )


def on_pre_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    conversation_history: Any = None,
    user_message: Any = None,
    **_: Any,
) -> None:
    """Capture the start-of-turn conversation so request-scoped hooks have an input.

    Hermes does not pass ``messages`` to ``pre_api_request``. ``pre_llm_call``
    is the only hook that receives the actual conversation as
    ``conversation_history`` (the message list at the start of the turn).
    We snapshot it here and extend in-place as the tool-calling loop runs.
    """
    if not isinstance(conversation_history, list):
        return
    convo = list(conversation_history)
    if isinstance(user_message, str) and user_message and not any(
        isinstance(m, dict) and m.get("role") == "user" and m.get("content") == user_message
        for m in convo
    ):
        convo.append({"role": "user", "content": user_message})
    _state.convo_set(_convo_key(task_id, session_id), convo)


def on_post_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    conversation_history: Any = None,
    assistant_response: Any = None,
    **_: Any,
) -> None:
    """Close all pending recorders for this turn with outputs from the final convo.

    ``conversation_history`` here is the FINAL state of the turn — includes
    every assistant message (with content/tool_calls) and every tool result.
    We pair the new assistant messages with our pending recorders in order.
    """
    _close_pending_for_session(session_id or "", conversation_history)
    _state.convo_clear(_convo_key(task_id, session_id))


def on_pre_api_request(
    *,
    task_id: str = "",
    session_id: str = "",
    model: str = "",
    provider: str = "",
    messages: Any = None,
    api_call_count: int = 0,
    **_: Any,
) -> None:
    client = _client._get_client()
    if client is None:
        return
    if not _should_sample():
        return

    try:
        from sigil_sdk import GenerationStart, ModelRef

        # Hermes does not pass ``messages`` to pre_api_request. Fall back to
        # the running history captured by ``pre_llm_call`` and extended by
        # post_api_request / post_tool_call.
        if not isinstance(messages, list):
            messages = _state.convo_get(_convo_key(task_id, session_id))
        system_prompt, non_system = _split_system_prompt(messages)
        sigil_messages = _messages_to_sigil(non_system)

        # Stamp started_at on both seed and GenState. The seed timestamp is
        # what the SDK uses for the span's start_time; GenState carries it so
        # the close path can compute completed_at = started_at + api_duration.
        started_at = datetime.now(UTC)
        start = GenerationStart(
            model=ModelRef(provider=provider or "unknown", name=model or "unknown"),
            conversation_id=session_id or task_id or "",
            agent_name=_agent_name(),
            system_prompt=system_prompt,
            started_at=started_at,
            tags={
                "sigil.framework.name": "hermes",
                "sigil.framework.source": "plugin",
                "sigil.framework.language": "python",
            },
            metadata={
                "hermes.api_call_count": api_call_count,
                "hermes.task_id": task_id,
                "hermes.session_id": session_id,
            },
        )
        recorder = client.start_generation(start)
        recorder.__enter__()
        # Input is stashed on GenState and threaded into set_result at
        # close-time — no need to call set_result twice.
        _state.gen_put(
            (task_id, session_id, int(api_call_count or 0)),
            _state.GenState(
                recorder=recorder,
                input_messages=sigil_messages,
                system_prompt=system_prompt,
                started_at=started_at,
            ),
        )
    except Exception as exc:
        logger.warning("hermes-plugin-sigil: on_pre_api_request failed: %s", exc)


def on_post_api_request(
    *,
    task_id: str = "",
    session_id: str = "",
    api_call_count: int = 0,
    model: str = "",
    usage: Any = None,
    finish_reason: str = "",
    response_model: str = "",
    api_duration: float | None = None,
    **_: Any,
) -> None:
    """Save partial result data on the GenState — recorder stays open.

    Hermes does not pass ``assistant_message`` to ``post_api_request`` (only
    ``assistant_content_chars`` and ``assistant_tool_call_count`` — see
    ``run_agent.py:12526``). The actual content of each call's assistant
    response is only available later via ``post_llm_call``'s
    ``conversation_history``. We defer ``set_result`` and recorder close
    until then. The assistant tool-call message that bridges this call to
    the next pre_api_request input chain is synthesized in post_tool_call,
    which is the only hook that reliably carries tool_name/args/tool_call_id
    on current hermes.
    """
    state = _state.gen_get((task_id, session_id, int(api_call_count or 0)))
    if state is None:
        return
    state.usage = usage
    state.finish_reason = finish_reason or ""
    state.response_model = response_model or model or ""
    if isinstance(api_duration, (int, float)) and api_duration >= 0:
        state.api_duration = float(api_duration)


def _close_pending_for_session(session_id: str, conversation_history: Any) -> None:
    """Drain pending generation recorders, assigning outputs from the final convo.

    Called from ``post_llm_call`` (normal path) and ``on_session_end``
    (interrupt safety). Walks the new portion of ``conversation_history`` to
    find assistant messages in order and pairs them with stored GenStates by
    api_call_count.
    """
    pending = _state.gen_pop_session(session_id or "")
    if not pending:
        return

    # Collect new assistant messages from conversation_history in order.
    asst_messages: list[dict] = []
    if isinstance(conversation_history, list):
        for msg in conversation_history:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                asst_messages.append(msg)
    # Take the last N assistant messages — those are this turn's outputs.
    # Pre-existing assistants from earlier turns appear first; pending list
    # length tells us how many to claim.
    relevant_asst = asst_messages[-len(pending):] if asst_messages else []

    for idx, ((_, _, _api_call_count), gen_state) in enumerate(pending):
        recorder = gen_state.recorder
        asst = relevant_asst[idx] if idx < len(relevant_asst) else None
        # Catch around the whole prep + set_result path: a malformed assistant
        # message or usage dict would otherwise abort the loop midway, leaving
        # later recorders open and skipping the caller's downstream
        # convo_clear / client.flush.
        try:
            sigil_output = _assistant_message_to_sigil(asst) if asst is not None else []
            token_usage = _build_token_usage(gen_state.usage)
            # Pin completed_at to started_at + api_duration when we have it.
            # The recorder is closed at post_llm_call time, which can be much
            # later than the LLM call itself (after tool execution + further
            # API calls in the same turn). Without this, the SDK's span end
            # and operation.duration histogram would cover the whole turn.
            completed_at: datetime | None = None
            if gen_state.started_at is not None and gen_state.api_duration is not None:
                completed_at = gen_state.started_at + timedelta(seconds=gen_state.api_duration)
            recorder.set_result(
                input=gen_state.input_messages,
                output=sigil_output,
                usage=token_usage,
                stop_reason=gen_state.finish_reason,
                response_model=gen_state.response_model,
                started_at=gen_state.started_at,
                completed_at=completed_at,
            )
        except Exception as exc:
            logger.warning("hermes-plugin-sigil: close-pending set_result failed: %s", exc)
        try:
            recorder.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("hermes-plugin-sigil: recorder __exit__ failed: %s", exc)


def on_post_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int | None = None,
    **_: Any,
) -> None:
    """Record the tool execution and extend the running convo for the next call.

    All work is done here, not split with pre_tool_call. Current hermes invokes
    pre_tool_call from ``run_agent.py:9060`` and ``run_agent.py:9520`` without
    ``session_id`` / ``tool_call_id`` (they default to ``""`` in
    ``get_pre_tool_call_block_message``), but post_tool_call (``model_tools.py:732``)
    always carries the real ids. Doing everything in post avoids a key mismatch
    between the two hooks that would leak recorders and misroute convo state.

    Order: append the synthesized assistant tool-call message, then the tool
    result, so the next ``pre_api_request``'s input chain reads
    ``user → assistant(tool_calls) → tool``. Then start, set_result, and close
    the tool execution recorder, using ``duration_ms`` from hermes to backdate
    the span's started_at so its duration reflects the tool's wallclock time.
    """
    convo_key = _convo_key(task_id, session_id)
    if tool_call_id:
        try:
            args_str = json.dumps(args) if args is not None else "{}"
        except Exception:
            args_str = "{}"
        _state.convo_append(
            convo_key,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name or "", "arguments": args_str},
                    }
                ],
            },
        )
        try:
            content = result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception:
            content = repr(result)
        _state.convo_append(
            convo_key,
            {"role": "tool", "tool_call_id": tool_call_id, "content": content},
        )

    client = _client._get_client()
    if client is None:
        return
    if not _should_sample():
        return

    try:
        from sigil_sdk import ToolExecutionStart

        completed_at = datetime.now(UTC)
        if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
            started_at = completed_at - timedelta(milliseconds=float(duration_ms))
        else:
            started_at = completed_at

        start = ToolExecutionStart(
            tool_name=tool_name or "",
            tool_call_id=tool_call_id or "",
            conversation_id=session_id or task_id or "",
            agent_name=_agent_name(),
            include_content=True,
            started_at=started_at,
        )
        recorder = client.start_tool_execution(start)
        recorder.__enter__()
        cfg = _client._get_plugin_config()
        max_chars = cfg.max_chars if cfg is not None else None
        try:
            try:
                recorder.set_result(
                    arguments=_redact.safe_value(args, max_chars=max_chars, parse_json_strings=True),
                    result=_redact.safe_value(result, max_chars=max_chars, parse_json_strings=True),
                    completed_at=completed_at,
                )
            except Exception as exc:
                logger.warning("hermes-plugin-sigil: tool set_result failed: %s", exc)
        finally:
            try:
                recorder.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("hermes-plugin-sigil: tool recorder __exit__ failed: %s", exc)
    except Exception as exc:
        logger.warning("hermes-plugin-sigil: on_post_tool_call failed: %s", exc)


def on_session_end(*, session_id: str = "", **_: Any) -> None:
    # Interrupt safety: if post_llm_call did not fire (e.g. user interrupted
    # or the turn errored out — see run_agent.py:13397 ``if final_response and
    # not interrupted``), pending recorders would leak. Close them here with
    # whatever partial state we have. ``conversation_history=None`` means we
    # cannot recover output content, but at least the recorders are closed
    # and partial state (input, usage) flows to Sigil.
    if session_id:
        _close_pending_for_session(session_id, None)

    # flush() leaves the singleton client open so subsequent hermes sessions
    # in the same process keep working. shutdown() would set _closed=True and
    # every future start_generation/start_tool_execution call would raise.
    client = _client._get_client(create_if_missing=False)
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            logger.warning("hermes-plugin-sigil: client.flush failed: %s", exc)
    # Flush the BatchSpanProcessor we installed — Client.flush() drains the
    # SDK's JSON export channel, but the OTel pipeline is independent.
    _client._flush_otel()
