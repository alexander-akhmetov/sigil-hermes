"""Verify register(ctx) binds exactly the five expected hooks."""
from __future__ import annotations

import hermes_plugin_sigil

EXPECTED_HOOKS = {
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "post_tool_call",
    "on_session_end",
}


def test_register_binds_exactly_expected_hooks(ctx) -> None:
    hermes_plugin_sigil.register(ctx)
    assert set(ctx.hooks.keys()) == EXPECTED_HOOKS


def test_register_handlers_are_callable(ctx) -> None:
    hermes_plugin_sigil.register(ctx)
    for name, handler in ctx.hooks.items():
        assert callable(handler), f"handler for {name} is not callable"


def test_register_does_not_bind_session_start(ctx) -> None:
    hermes_plugin_sigil.register(ctx)
    assert "on_session_start" not in ctx.hooks
