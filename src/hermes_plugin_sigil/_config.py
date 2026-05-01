"""Plugin-specific configuration for hermes-plugin-sigil.

Transport, auth, agent identity, debug, and content-capture-mode resolution
are owned by the Sigil SDK's ``Client()`` constructor — see the canonical
``SIGIL_*`` schema (``SIGIL_ENDPOINT``, ``SIGIL_PROTOCOL``, ``SIGIL_AUTH_*``,
``SIGIL_AGENT_NAME``, ``SIGIL_DEBUG``, ``SIGIL_CONTENT_CAPTURE_MODE``).

OTel exporter and resource resolution follow the standard OpenTelemetry env
schema (``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
``OTEL_SERVICE_NAME``, ``OTEL_RESOURCE_ATTRIBUTES``); the OTLP HTTP exporters
read these themselves.

This module only resolves plugin-specific knobs under the ``SIGIL_HERMES_*``
prefix (matching the ``SIGIL_CC_*`` / ``SIGIL_PI_*`` convention used by sibling
plugins) and tracks two presence flags driving channel decisions in
``_client.py`` and ``_otel.py``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SigilPluginConfig:
    """Resolved plugin-specific configuration."""

    sample_rate: float = 1.0
    max_chars: int = 12000
    otel_auto: bool = True
    generations_configured: bool = False
    otel_configured: bool = False


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("hermes-plugin-sigil: invalid %s=%r, using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("hermes-plugin-sigil: invalid %s=%r, using default %s", name, raw, default)
        return default


def _generations_configured() -> bool:
    if _env("SIGIL_AUTH_TOKEN"):
        return True
    mode = _env("SIGIL_AUTH_MODE").lower()
    return bool(mode) and mode != "none"


def _otel_configured() -> bool:
    return bool(_env("OTEL_EXPORTER_OTLP_ENDPOINT"))


def load() -> SigilPluginConfig:
    """Resolve plugin-specific env vars to a config.

    Always returns a config — channel decisions are driven by
    ``generations_configured`` / ``otel_configured`` rather than ``None``.
    """
    return SigilPluginConfig(
        sample_rate=_env_float("SIGIL_HERMES_SAMPLE_RATE", 1.0),
        max_chars=_env_int("SIGIL_HERMES_MAX_CHARS", 12000),
        otel_auto=_env_bool("SIGIL_HERMES_OTEL_AUTO", True),
        generations_configured=_generations_configured(),
        otel_configured=_otel_configured(),
    )
