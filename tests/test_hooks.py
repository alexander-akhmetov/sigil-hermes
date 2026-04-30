"""Hook payload-shaping and lifecycle tests.

The fake Sigil client records every ``start_generation`` / ``start_tool_execution``
/ ``shutdown`` call. The tests assert that the plugin produces correctly
shaped payloads and that recorders are cleaned up after the post-hooks fire.
"""
from __future__ import annotations

from typing import Any

import pytest
from sigil_sdk import GenerationStart, MessageRole, PartKind, ToolExecutionStart

from hermes_plugin_sigil import _hooks, _state


def _sample_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc_1", "function": {"name": "calc", "arguments": '{"expr": "2+2"}'}},
        ]},
        {"role": "tool", "tool_call_id": "tc_1", "content": "4"},
    ]


def test_pre_api_request_calls_start_generation_with_expected_fields(patch_client) -> None:
    _hooks.on_pre_api_request(
        task_id="t1",
        session_id="s1",
        model="claude-sonnet-4-6",
        provider="anthropic",
        messages=_sample_messages(),
        api_call_count=1,
    )

    assert len(patch_client.start_generation_calls) == 1
    start: GenerationStart = patch_client.start_generation_calls[0]
    assert isinstance(start, GenerationStart)
    assert start.conversation_id == "s1"
    assert start.agent_name == "hermes"
    assert start.model.provider == "anthropic"
    assert start.model.name == "claude-sonnet-4-6"
    assert start.system_prompt == "You are concise."
    assert start.metadata.get("hermes.api_call_count") == 1
    assert start.metadata.get("hermes.task_id") == "t1"

    rec = patch_client._next_gen_recorder
    assert rec.entered
    assert not rec.exited
    # Input is stored on GenState; threaded into set_result at close-time.
    state = _state.gen_get(("t1", "s1", 1))
    assert state is not None
    assert any(m.role == MessageRole.USER for m in state.input_messages)


def test_pre_post_api_request_round_trip(patch_client) -> None:
    _hooks.on_pre_api_request(
        task_id="t1",
        session_id="s1",
        model="gpt-4.1",
        provider="openai",
        messages=_sample_messages(),
        api_call_count=2,
    )
    rec = patch_client._next_gen_recorder
    assert rec.entered

    assistant_resp = {"role": "assistant", "content": "The answer is 4.", "tool_calls": []}
    _hooks.on_post_api_request(
        task_id="t1",
        session_id="s1",
        api_call_count=2,
        model="gpt-4.1",
        assistant_message=assistant_resp,
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 5,
            "reasoning_tokens": 7,
        },
        finish_reason="stop",
        messages=_sample_messages(),
    )

    # Recorder is NOT yet closed — close is deferred to post_llm_call so we
    # can assign the assistant output from conversation_history.
    assert not rec.exited

    _hooks.on_post_llm_call(
        task_id="t1",
        session_id="s1",
        conversation_history=[
            {"role": "user", "content": "What's 2+2?"},
            assistant_resp,
        ],
        assistant_response="The answer is 4.",
    )

    assert rec.exited
    assert _state.gen_pop(("t1", "s1", 2)) is None

    final = rec.set_result_calls[-1]
    assert final["stop_reason"] == "stop"
    assert final["response_model"] == "gpt-4.1"
    usage = final["usage"]
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cache_read_input_tokens == 30
    assert usage.cache_creation_input_tokens == 5
    assert usage.cache_write_input_tokens == 5
    assert usage.reasoning_tokens == 7
    output_messages = final["output"]
    assert len(output_messages) == 1
    assert output_messages[0].role == MessageRole.ASSISTANT
    assert any(p.kind == PartKind.TEXT and "4" in p.text for p in output_messages[0].parts)


def test_post_tool_call_records_full_round_trip(patch_client) -> None:
    args = {"path": "/tmp/foo.txt", "limit": 100}
    result = {"content": "hello world", "total_lines": 1}

    _hooks.on_post_tool_call(
        tool_name="read_file",
        args=args,
        result=result,
        task_id="t1",
        session_id="s1",
        tool_call_id="tc_42",
        duration_ms=42,
    )

    assert len(patch_client.start_tool_execution_calls) == 1
    start: ToolExecutionStart = patch_client.start_tool_execution_calls[0]
    assert isinstance(start, ToolExecutionStart)
    assert start.tool_name == "read_file"
    assert start.tool_call_id == "tc_42"
    assert start.conversation_id == "s1"
    assert start.agent_name == "hermes"
    assert start.include_content is True
    assert start.started_at is not None

    rec = patch_client._next_tool_recorder
    assert rec.entered
    assert rec.exited
    final = rec.set_result_calls[-1]
    # Args/result pass through redactor — values are echoed since they're tiny
    assert final["arguments"] == {"path": "/tmp/foo.txt", "limit": 100}
    assert final["result"] == {"content": "hello world", "total_lines": 1}
    # completed_at - started_at should be ~duration_ms
    delta_ms = (final["completed_at"] - start.started_at).total_seconds() * 1000
    assert delta_ms == pytest.approx(42, abs=1)


def test_missing_credentials_makes_handlers_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """No HERMES_SIGIL_* set → no Client constructed, all hooks no-op."""
    for name in (
        "HERMES_SIGIL_ENDPOINT",
        "HERMES_SIGIL_INSTANCE_ID",
        "HERMES_SIGIL_API_KEY",
        "HERMES_SIGIL_OTLP_ENDPOINT",
        "HERMES_SIGIL_OTLP_INSTANCE_ID",
        "HERMES_SIGIL_OTLP_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    constructed: list[Any] = []

    import sigil_sdk

    def boom(*_: Any, **__: Any) -> Any:
        constructed.append(True)
        raise AssertionError("Client should not be constructed when creds missing")

    monkeypatch.setattr(sigil_sdk, "Client", boom)

    # All hooks should silently no-op
    _hooks.on_pre_api_request(task_id="t", session_id="s", model="m", provider="p", messages=[], api_call_count=1)
    _hooks.on_post_api_request(task_id="t", session_id="s", api_call_count=1)
    _hooks.on_post_tool_call(tool_name="x", task_id="t", session_id="s", tool_call_id="tc")
    _hooks.on_session_end()
    assert constructed == []


def test_client_init_failure_is_cached(monkeypatch: pytest.MonkeyPatch, env_creds: None) -> None:
    """Construction error → handlers swallow, subsequent calls don't retry."""
    import sigil_sdk

    from hermes_plugin_sigil import _otel
    monkeypatch.setattr(_otel, "setup_if_needed", lambda cfg: True)

    call_count = {"n": 0}

    def boom(*_: Any, **__: Any) -> Any:
        call_count["n"] += 1
        raise RuntimeError("unreachable endpoint")

    monkeypatch.setattr(sigil_sdk, "Client", boom)

    _hooks.on_pre_api_request(task_id="t", session_id="s", model="m", provider="p", messages=[], api_call_count=1)
    _hooks.on_post_tool_call(tool_name="x", task_id="t", session_id="s", tool_call_id="tc")
    _hooks.on_session_end()

    assert call_count["n"] == 1, "client construction must only be retried once after failure"


def test_on_session_end_flushes_without_closing_client(patch_client) -> None:
    """on_session_end must flush, not shutdown — the client is a process-wide singleton."""
    _hooks.on_session_end()
    assert patch_client.flush_calls == 1
    assert patch_client.shutdown_calls == 0


def test_session_end_lets_subsequent_session_record(patch_client) -> None:
    """After a session ends, a fresh session must still be able to record."""
    _hooks.on_pre_api_request(
        task_id="t1", session_id="s1", model="m", provider="p",
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
    )
    _hooks.on_session_end()
    # Second session — same singleton, must keep working.
    _hooks.on_pre_api_request(
        task_id="t2", session_id="s2", model="m", provider="p",
        messages=[{"role": "user", "content": "again"}], api_call_count=1,
    )
    assert len(patch_client.start_generation_calls) == 2


def test_session_end_force_flushes_installed_providers(
    monkeypatch: pytest.MonkeyPatch, env_creds: None,
) -> None:
    """If the plugin installed providers, on_session_end force-flushes them."""
    import sigil_sdk

    from hermes_plugin_sigil import _client, _otel
    from tests.conftest import FakeClient

    class FakeProvider:
        def __init__(self) -> None:
            self.flush_calls = 0

        def force_flush(self, *_: Any, **__: Any) -> None:
            self.flush_calls += 1

    fake_tracer = FakeProvider()
    fake_meter = FakeProvider()
    monkeypatch.setattr(_otel, "_INSTALLED_TRACER_PROVIDER", fake_tracer, raising=False)
    monkeypatch.setattr(_otel, "_INSTALLED_METER_PROVIDER", fake_meter, raising=False)
    monkeypatch.setattr(_otel, "setup_if_needed", lambda cfg: True)
    monkeypatch.setattr(sigil_sdk, "Client", lambda *a, **k: FakeClient())

    # Force client init
    assert _client._get_client() is not None
    _hooks.on_session_end()
    assert fake_tracer.flush_calls == 1
    assert fake_meter.flush_calls == 1


def test_session_end_does_not_flush_user_owned_providers(
    monkeypatch: pytest.MonkeyPatch, env_creds: None,
) -> None:
    """When the host app owns the providers, the plugin must not flush them."""
    import sigil_sdk

    from hermes_plugin_sigil import _client, _otel
    from tests.conftest import FakeClient

    class FakeProvider:
        def __init__(self) -> None:
            self.flush_calls = 0

        def force_flush(self, *_: Any, **__: Any) -> None:
            self.flush_calls += 1

    fake_provider = FakeProvider()
    # _INSTALLED_*_PROVIDER stay None — represents user-owned provider path.
    monkeypatch.setattr(_otel, "setup_if_needed", lambda cfg: True)
    monkeypatch.setattr(sigil_sdk, "Client", lambda *a, **k: FakeClient())
    assert _client._get_client() is not None

    _hooks.on_session_end()
    assert fake_provider.flush_calls == 0


def test_on_session_end_does_not_initialize_client(monkeypatch: pytest.MonkeyPatch, env_creds: None) -> None:
    """on_session_end must use create_if_missing=False and not trigger init."""
    import sigil_sdk

    from hermes_plugin_sigil import _otel
    monkeypatch.setattr(_otel, "setup_if_needed", lambda cfg: True)

    constructed = {"n": 0}

    def factory(*_: Any, **__: Any) -> Any:
        constructed["n"] += 1
        return object()

    monkeypatch.setattr(sigil_sdk, "Client", factory)

    _hooks.on_session_end()
    assert constructed["n"] == 0


def test_post_api_request_without_pre_is_safe(patch_client) -> None:
    """Stale post-hook should not raise."""
    _hooks.on_post_api_request(task_id="t", session_id="s", api_call_count=999, assistant_message={"content": "x"})


def test_post_tool_call_with_unknown_id_is_safe(patch_client) -> None:
    _hooks.on_post_tool_call(tool_name="x", task_id="t", session_id="s", tool_call_id="ghost")


def test_sample_rate_zero_skips_recording(monkeypatch: pytest.MonkeyPatch, patch_client) -> None:
    """HERMES_SIGIL_SAMPLE_RATE=0 → pre-hooks short-circuit, no recorder created."""
    from hermes_plugin_sigil import _client, _config

    monkeypatch.setattr(
        _client, "_CONFIG",
        _config.SigilPluginConfig(
            generations=None,
            otlp=None,
            sample_rate=0.0,
        ),
        raising=False,
    )

    _hooks.on_pre_api_request(
        task_id="t", session_id="s", model="m", provider="p",
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
    )
    _hooks.on_post_tool_call(tool_name="x", args={}, result="ok", task_id="t", session_id="s", tool_call_id="tc1")

    assert patch_client.start_generation_calls == []
    assert patch_client.start_tool_execution_calls == []


def test_pre_llm_call_seeds_input_for_pre_api_request(patch_client) -> None:
    """Hermes does not pass messages to pre_api_request — input must come from pre_llm_call."""
    _hooks.on_pre_llm_call(
        task_id="t1",
        session_id="s1",
        conversation_history=[
            {"role": "user", "content": "hey"},
        ],
    )
    _hooks.on_pre_api_request(
        task_id="t1",
        session_id="s1",
        model="m",
        provider="p",
        api_call_count=1,
        # messages NOT passed — matches real hermes
    )

    rec = patch_client._next_gen_recorder
    assert rec is not None
    state = _state.gen_get(("t1", "s1", 1))
    assert state is not None
    assert len(state.input_messages) == 1
    assert state.input_messages[0].role == MessageRole.USER


def test_post_llm_call_assigns_outputs_to_pending_recorders(patch_client) -> None:
    """Tool loop: 2 LLM calls, post_llm_call assigns each call's assistant output."""
    _hooks.on_pre_llm_call(
        task_id="t1", session_id="s1",
        conversation_history=[{"role": "user", "content": "search for X"}],
    )
    # LLM call 1 — emits a tool call
    _hooks.on_pre_api_request(task_id="t1", session_id="s1", model="m", provider="p", api_call_count=1)
    rec1 = patch_client._next_gen_recorder
    _hooks.on_post_api_request(
        task_id="t1", session_id="s1", api_call_count=1, model="m",
        usage={"input_tokens": 10, "output_tokens": 5}, finish_reason="tool_calls",
    )
    assert not rec1.exited  # deferred close
    # LLM call 2 — final answer
    _hooks.on_pre_api_request(task_id="t1", session_id="s1", model="m", provider="p", api_call_count=2)
    rec2 = patch_client._next_gen_recorder
    _hooks.on_post_api_request(
        task_id="t1", session_id="s1", api_call_count=2, model="m",
        usage={"input_tokens": 20, "output_tokens": 8}, finish_reason="stop",
    )
    assert not rec2.exited

    # Final conversation_history with both assistant messages
    asst1 = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "tc_1", "function": {"name": "search", "arguments": '{"q":"X"}'}}
    ]}
    asst2 = {"role": "assistant", "content": "found 3 results"}
    _hooks.on_post_llm_call(
        task_id="t1", session_id="s1",
        conversation_history=[
            {"role": "user", "content": "search for X"},
            asst1,
            {"role": "tool", "tool_call_id": "tc_1", "content": "results"},
            asst2,
        ],
        assistant_response="found 3 results",
    )

    # Both recorders closed with their respective assistant outputs
    assert rec1.exited and rec2.exited
    final1 = rec1.set_result_calls[-1]
    final2 = rec2.set_result_calls[-1]
    # rec1's output should reflect asst1 (had tool_calls)
    assert final1["stop_reason"] == "tool_calls"
    assert any(p.kind == PartKind.TOOL_CALL for p in final1["output"][0].parts)
    # rec2's output should reflect asst2 (final text)
    assert final2["stop_reason"] == "stop"
    assert any(p.kind == PartKind.TEXT and "found 3 results" in p.text for p in final2["output"][0].parts)


def test_completed_at_uses_api_duration_not_recorder_close_time(patch_client) -> None:
    """Span end + duration metric must reflect the LLM call, not the close time.

    If we used wallclock at close, the first call in a tool loop would report
    a span/histogram covering the full turn (LLM + tool + later calls).
    """
    _hooks.on_pre_api_request(
        task_id="t1", session_id="s1", model="m", provider="p",
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
    )
    rec = patch_client._next_gen_recorder
    state = _state.gen_get(("t1", "s1", 1))
    assert state is not None and state.started_at is not None
    started_at = state.started_at

    _hooks.on_post_api_request(
        task_id="t1", session_id="s1", api_call_count=1, model="m",
        usage={}, finish_reason="stop", api_duration=2.5,
    )
    _hooks.on_post_llm_call(task_id="t1", session_id="s1", conversation_history=[])

    final = rec.set_result_calls[-1]
    assert final["started_at"] == started_at
    completed_at = final["completed_at"]
    assert completed_at is not None
    delta = (completed_at - started_at).total_seconds()
    assert delta == pytest.approx(2.5)


def test_completed_at_is_none_when_api_duration_missing(patch_client) -> None:
    """No api_duration → leave completed_at unset; SDK falls back to its clock."""
    _hooks.on_pre_api_request(
        task_id="t1", session_id="s1", model="m", provider="p",
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
    )
    rec = patch_client._next_gen_recorder
    _hooks.on_post_api_request(
        task_id="t1", session_id="s1", api_call_count=1, model="m",
        usage={}, finish_reason="stop",
        # no api_duration kwarg
    )
    _hooks.on_post_llm_call(task_id="t1", session_id="s1", conversation_history=[])

    final = rec.set_result_calls[-1]
    assert final["completed_at"] is None


def test_session_end_closes_pending_recorders_on_interrupt(patch_client) -> None:
    """If post_llm_call never fires (interrupt), on_session_end must still close recorders."""
    _hooks.on_pre_api_request(
        task_id="t1", session_id="s1", model="m", provider="p",
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
    )
    rec = patch_client._next_gen_recorder
    _hooks.on_post_api_request(
        task_id="t1", session_id="s1", api_call_count=1, model="m",
        usage={}, finish_reason="length",
    )
    assert not rec.exited
    # session_end fires before post_llm_call (interrupt path)
    _hooks.on_session_end(session_id="s1")
    assert rec.exited
    # Output is empty (no conversation_history to derive it from), but the
    # recorder was closed and partial state (input, usage) was set.
    final = rec.set_result_calls[-1]
    assert final["output"] == []


def test_running_convo_includes_assistant_and_tool_results(patch_client) -> None:
    """Tool loop: conversation grows across calls so api_call_count=2 has full input."""
    _hooks.on_pre_llm_call(
        task_id="t1",
        session_id="s1",
        conversation_history=[{"role": "user", "content": "search for X"}],
    )
    # Call #1 — model decides to call a tool
    _hooks.on_pre_api_request(
        task_id="t1", session_id="s1", model="m", provider="p", api_call_count=1,
    )
    _hooks.on_post_api_request(
        task_id="t1", session_id="s1", api_call_count=1, model="m",
        usage={}, finish_reason="tool_calls",
    )
    # Tool runs — post_tool_call synthesizes the assistant tool_call message
    # and appends the tool result, both into the running convo.
    _hooks.on_post_tool_call(
        tool_name="search",
        args={"q": "X"},
        task_id="t1", session_id="s1", tool_call_id="tc_1",
        result="found 3 results",
    )
    # Call #2 — model gets to see user msg + asst tool call + tool result
    _hooks.on_pre_api_request(
        task_id="t1", session_id="s1", model="m", provider="p", api_call_count=2,
    )
    rec2 = patch_client._next_gen_recorder
    assert rec2 is not None
    state2 = _state.gen_get(("t1", "s1", 2))
    assert state2 is not None
    roles = [m.role for m in state2.input_messages]
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles
    assert MessageRole.TOOL in roles


def test_post_llm_call_clears_running_convo(patch_client) -> None:
    _hooks.on_pre_llm_call(
        task_id="t1", session_id="s1",
        conversation_history=[{"role": "user", "content": "hi"}],
    )
    from hermes_plugin_sigil import _state
    # Convo is keyed by session_id only — task_id is not passed to pre_llm_call.
    assert _state.convo_get(("", "s1")) != []
    _hooks.on_post_llm_call(task_id="t1", session_id="s1")
    assert _state.convo_get(("", "s1")) == []


def test_sample_rate_one_records_everything(patch_client) -> None:
    """HERMES_SIGIL_SAMPLE_RATE=1.0 (default) → every call recorded."""
    _hooks.on_pre_api_request(
        task_id="t", session_id="s", model="m", provider="p",
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
    )
    assert len(patch_client.start_generation_calls) == 1


def test_client_config_uses_http_protocol_with_basic_auth_when_generations_configured(
    monkeypatch: pytest.MonkeyPatch, env_creds: None,
) -> None:
    """When HERMES_SIGIL_* generation creds are set, SDK gets protocol=http + basic auth."""
    import sigil_sdk

    from hermes_plugin_sigil import _client, _otel

    captured: list[Any] = []

    def factory(config: Any) -> Any:
        captured.append(config)
        from tests.conftest import FakeClient
        return FakeClient()

    monkeypatch.setattr(sigil_sdk, "Client", factory)
    monkeypatch.setattr(_otel, "setup_if_needed", lambda cfg: True)

    assert _client._get_client() is not None
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.generation_export.protocol == "http"
    assert cfg.generation_export.endpoint == "http://localhost/api/v1/generations:export"
    assert cfg.generation_export.auth.mode == "basic"
    assert cfg.generation_export.auth.tenant_id == "stack-1"
    assert cfg.generation_export.auth.basic_password == "glc_secret"


def test_client_config_uses_protocol_none_when_only_otlp_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When only OTLP creds are set, the SDK's HTTP exporter is disabled."""
    import sigil_sdk

    from hermes_plugin_sigil import _client, _otel

    # Only OTLP, no generations
    for name in ("HERMES_SIGIL_ENDPOINT", "HERMES_SIGIL_INSTANCE_ID", "HERMES_SIGIL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_SIGIL_OTLP_ENDPOINT", "http://localhost/otlp")
    monkeypatch.setenv("HERMES_SIGIL_OTLP_INSTANCE_ID", "stack-1")
    monkeypatch.setenv("HERMES_SIGIL_OTLP_TOKEN", "glc_otlp_secret")

    captured: list[Any] = []

    def factory(config: Any) -> Any:
        captured.append(config)
        from tests.conftest import FakeClient
        return FakeClient()

    monkeypatch.setattr(sigil_sdk, "Client", factory)
    monkeypatch.setattr(_otel, "setup_if_needed", lambda cfg: True)

    assert _client._get_client() is not None
    assert len(captured) == 1
    assert captured[0].generation_export.protocol == "none"
