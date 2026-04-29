"""Lazy Sigil client construction.

The client is built on first hook invocation. If no channel is configured
(no Sigil generation creds and no OTLP creds), the plugin is fully no-op.
If construction fails, the failure is cached — handlers never retry.

The Sigil SDK's HTTP generation exporter handles its own auth and batching.
OTel TracerProvider + MeterProvider are installed by ``_otel`` when the host
hasn't done it.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from . import _config, _otel

logger = logging.getLogger(__name__)

_INIT_FAILED = object()
_CLIENT: Any = None
_CONFIG: _config.SigilPluginConfig | None = None
_LOCK = threading.Lock()


def _to_sigil_client_config(cfg: _config.SigilPluginConfig):
    """Build a ``ClientConfig`` from the plugin config.

    When generation creds are absent, the JSON export channel is set to
    ``protocol="none"`` so the SDK doesn't try to dial localhost. When they
    are present, the SDK's HTTP exporter handles auth + retry + batching.
    """
    from sigil_sdk import AuthConfig, ClientConfig, ContentCaptureMode, GenerationExportConfig

    capture = ContentCaptureMode(cfg.content_capture)

    if cfg.generations is None:
        return ClientConfig(
            generation_export=GenerationExportConfig(protocol="none"),
            content_capture=capture,
        )

    return ClientConfig(
        generation_export=GenerationExportConfig(
            protocol="http",
            endpoint=cfg.generations.endpoint,
            auth=AuthConfig(
                mode="basic",
                tenant_id=cfg.generations.instance_id,
                basic_password=cfg.generations.api_key,
            ),
        ),
        content_capture=capture,
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
        if cfg is None:
            logger.warning(
                "hermes-plugin-sigil: no channel configured — set HERMES_SIGIL_ENDPOINT/INSTANCE_ID/API_KEY "
                "for generations or HERMES_SIGIL_OTLP_ENDPOINT/INSTANCE_ID/TOKEN for traces+metrics. "
                "Telemetry disabled."
            )
            _CLIENT = _INIT_FAILED
            return None

        # OTel setup is independent of the Sigil client — fine if it returns
        # False, generations channel can still work.
        _otel.setup_if_needed(cfg)

        try:
            from sigil_sdk import Client

            _CLIENT = Client(_to_sigil_client_config(cfg))
            _CONFIG = cfg
            if cfg.debug:
                logger.info(
                    "hermes-plugin-sigil: Sigil client initialized "
                    "(generations=%s, otlp=%s)",
                    "on" if cfg.generations else "off",
                    "on" if cfg.otlp else "off",
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
