"""hermes-plugin-sigil — Grafana AI Observability plugin for Hermes Agent.

Records every LLM API call (`pre_api_request`/`post_api_request`) as a Sigil
generation and every tool invocation (`pre_tool_call`/`post_tool_call`) as a
Sigil tool execution. On `on_session_end`, flushes the SDK's HTTP exporter
and any OTel providers the plugin installed.

Two independent channels — see README:
  - Generations:  HERMES_SIGIL_ENDPOINT / _INSTANCE_ID / _API_KEY
  - OTel (traces + metrics):  HERMES_SIGIL_OTLP_ENDPOINT / _INSTANCE_ID / _TOKEN

The plugin fails open: missing credentials, SDK errors, exporter failures, and
network errors all become silent no-ops after at most one warning log. The
hermes agent loop is never blocked or interrupted by telemetry issues.
"""
from __future__ import annotations

from ._hooks import (
    on_post_api_request,
    on_post_llm_call,
    on_post_tool_call,
    on_pre_api_request,
    on_pre_llm_call,
    on_pre_tool_call,
    on_session_end,
)


def register(ctx) -> None:
    """Register sigil hook handlers with the hermes plugin context."""
    # pre_llm_call / post_llm_call are turn-scoped — they bracket the running
    # conversation history that we use as input for each request-scoped
    # pre_api_request (which does not receive ``messages`` from hermes).
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_end", on_session_end)


__all__ = [
    "register",
    "on_pre_llm_call",
    "on_post_llm_call",
    "on_pre_api_request",
    "on_post_api_request",
    "on_pre_tool_call",
    "on_post_tool_call",
    "on_session_end",
]
