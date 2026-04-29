"""Test fixtures for hermes-plugin-sigil.

Each test gets a fake Sigil ``Client`` that records calls without touching the
network. The ``patch_client`` fixture replaces ``sigil_sdk.Client`` so that
``hermes_plugin_sigil._client._get_client()`` returns the fake. Module-level
state in ``_client``, ``_otel``, and ``_state`` is reset between tests so
ordering doesn't leak.
"""
from __future__ import annotations

from typing import Any

import pytest

from hermes_plugin_sigil import _client, _state


class FakeRecorder:
    """Records lifecycle calls for assertions."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.set_result_calls: list[dict[str, Any]] = []
        self.set_call_error_calls: list[Exception] = []

    def __enter__(self) -> FakeRecorder:
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.exited = True
        return False

    def set_result(self, *args: Any, **kwargs: Any) -> None:
        self.set_result_calls.append(dict(kwargs))

    def set_call_error(self, error: Exception) -> None:
        self.set_call_error_calls.append(error)


class FakeClient:
    """In-memory stand-in for ``sigil_sdk.Client``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.start_generation_calls: list[Any] = []
        self.start_tool_execution_calls: list[Any] = []
        self.flush_calls = 0
        self.shutdown_calls = 0
        self.init_args = args
        self.init_kwargs = kwargs
        self._next_gen_recorder: FakeRecorder | None = None
        self._next_tool_recorder: FakeRecorder | None = None

    def start_generation(self, start: Any) -> FakeRecorder:
        self.start_generation_calls.append(start)
        rec = FakeRecorder()
        self._next_gen_recorder = rec
        return rec

    def start_tool_execution(self, start: Any) -> FakeRecorder:
        self.start_tool_execution_calls.append(start)
        rec = FakeRecorder()
        self._next_tool_recorder = rec
        return rec

    def flush(self) -> None:
        self.flush_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture(autouse=True)
def reset_module_state() -> None:
    """Clear cached client + recorder state before every test."""
    _client._reset_for_tests()
    _state.reset_for_tests()
    yield
    _client._reset_for_tests()
    _state.reset_for_tests()


@pytest.fixture
def env_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both channels configured."""
    # Generations channel
    monkeypatch.setenv("HERMES_SIGIL_ENDPOINT", "http://localhost/api/v1/generations:export")
    monkeypatch.setenv("HERMES_SIGIL_INSTANCE_ID", "stack-1")
    monkeypatch.setenv("HERMES_SIGIL_API_KEY", "glc_secret")
    # OTel channel
    monkeypatch.setenv("HERMES_SIGIL_OTLP_ENDPOINT", "http://localhost/otlp")
    monkeypatch.setenv("HERMES_SIGIL_OTLP_INSTANCE_ID", "stack-1")
    monkeypatch.setenv("HERMES_SIGIL_OTLP_TOKEN", "glc_otlp_secret")


@pytest.fixture
def patch_client(monkeypatch: pytest.MonkeyPatch, env_creds: None) -> FakeClient:
    """Replace ``sigil_sdk.Client`` with ``FakeClient`` and skip OTel setup."""
    import sigil_sdk

    instances: list[FakeClient] = []

    def factory(*args: Any, **kwargs: Any) -> FakeClient:
        instance = FakeClient(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(sigil_sdk, "Client", factory)
    # Skip the real OTel auto-setup — tests for that path are isolated.
    from hermes_plugin_sigil import _otel
    monkeypatch.setattr(_otel, "setup_if_needed", lambda cfg: True)

    # Force lazy init by calling once
    client = _client._get_client()
    assert isinstance(client, FakeClient), "fake client should have been constructed"
    return client


class FakeContext:
    """Stand-in for the hermes plugin context object passed to ``register``."""

    def __init__(self) -> None:
        self.hooks: dict[str, Any] = {}

    def register_hook(self, name: str, handler: Any) -> None:
        self.hooks[name] = handler


@pytest.fixture
def ctx() -> FakeContext:
    return FakeContext()
