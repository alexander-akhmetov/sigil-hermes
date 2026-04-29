"""Environment-driven configuration for hermes-plugin-sigil.

Two independent channels — each has its own endpoint and basic-auth pair, and
each can be configured separately. If only one channel's vars are set, only
that channel runs.

Generations (Sigil API):
    HERMES_SIGIL_ENDPOINT       — e.g. https://<region>.grafana.net/api/v1/generations:export
    HERMES_SIGIL_INSTANCE_ID    — basic-auth user
    HERMES_SIGIL_API_KEY        — basic-auth password (sigil:write scope)

OTel (traces + metrics, via Grafana Cloud OTLP gateway):
    HERMES_SIGIL_OTLP_ENDPOINT      — e.g. https://otlp-gateway-prod-<region>.grafana.net/otlp
    HERMES_SIGIL_OTLP_INSTANCE_ID   — basic-auth user
    HERMES_SIGIL_OTLP_TOKEN         — basic-auth password (cloud-write scope)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationsConfig:
    endpoint: str
    instance_id: str
    api_key: str


@dataclass(slots=True)
class OTLPConfig:
    endpoint: str       # base URL — /v1/traces and /v1/metrics are appended per signal
    instance_id: str
    token: str


@dataclass(slots=True)
class SigilPluginConfig:
    """Resolved plugin configuration."""

    generations: GenerationsConfig | None
    otlp: OTLPConfig | None
    agent_name: str = "hermes"
    sample_rate: float = 1.0
    max_chars: int = 12000
    debug: bool = False
    otel_auto: bool = True
    # ``full`` by default — for an agent observability plugin you want tool
    # call args and results visible. The SDK's own default is no_tool_content,
    # which renders agent conversations as empty shells in the UI. Override
    # via HERMES_SIGIL_CONTENT_CAPTURE.
    content_capture: str = "full"

    @property
    def any_channel_configured(self) -> bool:
        return self.generations is not None or self.otlp is not None


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


def _load_generations() -> GenerationsConfig | None:
    endpoint = _env("HERMES_SIGIL_ENDPOINT")
    instance_id = _env("HERMES_SIGIL_INSTANCE_ID")
    api_key = _env("HERMES_SIGIL_API_KEY")
    present = sum(bool(v) for v in (endpoint, instance_id, api_key))
    if present == 0:
        return None
    if present < 3:
        logger.warning(
            "hermes-plugin-sigil: generations channel partially configured "
            "(need HERMES_SIGIL_ENDPOINT, HERMES_SIGIL_INSTANCE_ID, HERMES_SIGIL_API_KEY) — disabled"
        )
        return None
    return GenerationsConfig(endpoint=endpoint, instance_id=instance_id, api_key=api_key)


def _load_otlp() -> OTLPConfig | None:
    endpoint = _env("HERMES_SIGIL_OTLP_ENDPOINT")
    instance_id = _env("HERMES_SIGIL_OTLP_INSTANCE_ID")
    token = _env("HERMES_SIGIL_OTLP_TOKEN")
    present = sum(bool(v) for v in (endpoint, instance_id, token))
    if present == 0:
        return None
    if present < 3:
        logger.warning(
            "hermes-plugin-sigil: OTLP channel partially configured "
            "(need HERMES_SIGIL_OTLP_ENDPOINT, HERMES_SIGIL_OTLP_INSTANCE_ID, HERMES_SIGIL_OTLP_TOKEN) — disabled"
        )
        return None
    return OTLPConfig(endpoint=endpoint.rstrip("/"), instance_id=instance_id, token=token)


def load() -> SigilPluginConfig | None:
    """Resolve env vars to a config, or ``None`` if no channel is configured."""
    generations = _load_generations()
    otlp = _load_otlp()
    if generations is None and otlp is None:
        return None

    raw_capture = _env("HERMES_SIGIL_CONTENT_CAPTURE").lower()
    if raw_capture and raw_capture not in {"default", "full", "no_tool_content", "metadata_only"}:
        logger.warning(
            "hermes-plugin-sigil: invalid HERMES_SIGIL_CONTENT_CAPTURE=%r — using 'full'",
            raw_capture,
        )
        raw_capture = ""

    return SigilPluginConfig(
        generations=generations,
        otlp=otlp,
        agent_name=_env("HERMES_SIGIL_AGENT_NAME") or "hermes",
        sample_rate=_env_float("HERMES_SIGIL_SAMPLE_RATE", 1.0),
        max_chars=_env_int("HERMES_SIGIL_MAX_CHARS", 12000),
        debug=_env_bool("HERMES_SIGIL_DEBUG", False),
        otel_auto=_env_bool("HERMES_SIGIL_OTEL_AUTO", True),
        content_capture=raw_capture or "full",
    )
