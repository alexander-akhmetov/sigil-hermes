"""Lazy Sigil client construction.

The client is built on first hook invocation. If neither the generations nor
the OTel channel is configured, the plugin is fully no-op. If construction
fails, the failure is cached — handlers never retry.

The Sigil SDK's ``Client()`` constructor reads canonical ``SIGIL_*`` env vars
itself, so most callers pass no explicit config. The plugin supplies an
override only in OTel-only mode (no generations creds) to suppress the SDK's
HTTP exporter, and to default ``content_capture`` to ``full`` when the user
hasn't picked a mode (overriding the SDK's ``no_tool_content`` default which
hides tool I/O in the UI).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from . import _config, _otel

logger = logging.getLogger(__name__)

_INIT_FAILED = object()
_CLIENT: Any = None
_CONFIG: _config.SigilPluginConfig | None = None
_LOCK = threading.Lock()


def _to_sigil_client_config(cfg: _config.SigilPluginConfig):
    """Build an override ``ClientConfig`` for the SDK, or ``None`` for full env-resolution.

    When generation creds are present, returns ``None`` so ``Client()`` resolves
    everything from canonical env. In OTel-only mode returns a config that
    pins ``protocol="none"`` so the SDK's HTTP exporter doesn't dial the
    default ingest endpoint.

    Either way, ``content_capture=full`` is set explicitly when
    ``SIGIL_CONTENT_CAPTURE_MODE`` is unset — the SDK's default is
    ``no_tool_content``, which renders agent UIs empty for hermes traffic.
    """
    from sigil_sdk import ClientConfig, ContentCaptureMode, GenerationExportConfig

    overrides: dict[str, Any] = {}
    if not os.environ.get("SIGIL_CONTENT_CAPTURE_MODE"):
        overrides["content_capture"] = ContentCaptureMode.FULL

    if cfg.generations_configured:
        if not overrides:
            return None
        return ClientConfig(**overrides)

    return ClientConfig(
        generation_export=GenerationExportConfig(protocol="none"),
        **overrides,
    )


def _get_client(create_if_missing: bool = True) -> Any:
    """Return the cached Sigil client or ``None`` if init has failed/cannot run."""
    global _CLIENT, _CONFIG

    if _CLIENT is _INIT_FAILED:
        return None
    if _CLIENT is not None:
        return _CLIENT
    if not create_if_missing:
        return None

    with _LOCK:
        if _CLIENT is _INIT_FAILED:
            return None
        if _CLIENT is not None:
            return _CLIENT

        cfg = _config.load()
        if not (cfg.generations_configured or cfg.otel_configured):
            logger.warning(
                "hermes-plugin-sigil: no channel configured — set SIGIL_AUTH_TOKEN "
                "(with SIGIL_ENDPOINT/SIGIL_PROTOCOL/SIGIL_AUTH_*) for generations, "
                "or OTEL_EXPORTER_OTLP_ENDPOINT for traces+metrics. Telemetry disabled."
            )
            _CLIENT = _INIT_FAILED
            return None

        # OTel setup is independent of the Sigil client — fine if it returns
        # False, generations channel can still work.
        _otel.setup_if_needed(cfg)

        try:
            from sigil_sdk import Client

            override = _to_sigil_client_config(cfg)
            _CLIENT = Client() if override is None else Client(override)
            _CONFIG = cfg
            logger.info(
                "hermes-plugin-sigil: Sigil client initialized "
                "(generations=%s, otel=%s)",
                "configured" if cfg.generations_configured else "unconfigured",
                "configured" if cfg.otel_configured else "unconfigured",
            )
            return _CLIENT
        except Exception as exc:
            logger.warning("hermes-plugin-sigil: failed to initialize Sigil client: %s", exc)
            _CLIENT = _INIT_FAILED
            return None


def _get_plugin_config() -> _config.SigilPluginConfig | None:
    """Return the resolved plugin config, or ``None`` if the client never initialized."""
    return _CONFIG


def _flush_otel() -> None:
    """Force-flush OTel providers we installed. No-op for host-owned providers."""
    _otel.force_flush()


def _reset_for_tests() -> None:
    """Reset cached client/config state. Test-only."""
    global _CLIENT, _CONFIG
    with _LOCK:
        _CLIENT = None
        _CONFIG = None
    _otel._reset_for_tests()
