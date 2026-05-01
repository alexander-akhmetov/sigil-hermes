"""OpenTelemetry TracerProvider + MeterProvider auto-setup.

The Sigil SDK does not own OTel — applications must install providers. This
plugin acts as the application setup for hermes users who haven't wired OTel
themselves.

OTel exporter and resource configuration follow the OpenTelemetry env-var
schema (``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
``OTEL_SERVICE_NAME``, ``OTEL_RESOURCE_ATTRIBUTES``). The OTLP HTTP exporters
read these themselves; the plugin only fills in ``service.name=hermes`` when
the user hasn't set one.

If the host application has already installed a non-proxy provider, the plugin
leaves it untouched and uses the host's setup.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from . import _config

logger = logging.getLogger(__name__)

_INSTALLED_TRACER_PROVIDER: Any = None
_INSTALLED_METER_PROVIDER: Any = None
_SETUP_DONE = False


def _is_proxy_tracer_provider(provider: Any) -> bool:
    from opentelemetry.trace import ProxyTracerProvider

    return isinstance(provider, ProxyTracerProvider)


def _is_proxy_meter_provider(provider: Any) -> bool:
    # Public default is the proxy provider. The class is in `_internal` —
    # reaching into a private name is brittle but the only correct check.
    try:
        from opentelemetry.metrics._internal import _ProxyMeterProvider
    except ImportError:
        return False
    return isinstance(provider, _ProxyMeterProvider)


def _build_resource():
    from opentelemetry.sdk.resources import Resource

    # Resource.create() merges OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES
    # from env, but explicit attrs win on key collision. Only set service.name
    # when the user hasn't, so OTEL_SERVICE_NAME still takes effect.
    attrs: dict[str, str] = {}
    if not os.environ.get("OTEL_SERVICE_NAME") and "service.name=" not in os.environ.get(
        "OTEL_RESOURCE_ATTRIBUTES", ""
    ):
        attrs["service.name"] = "hermes"
    return Resource.create(attrs)


def _install_tracer_provider() -> Any:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # OTLPSpanExporter() reads OTEL_EXPORTER_OTLP_ENDPOINT (appending
    # /v1/traces), OTEL_EXPORTER_OTLP_HEADERS, and OTEL_EXPORTER_OTLP_INSECURE
    # itself.
    exporter = OTLPSpanExporter()
    provider = TracerProvider(resource=_build_resource())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _install_meter_provider() -> Any:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    exporter = OTLPMetricExporter()
    reader = PeriodicExportingMetricReader(exporter)
    return MeterProvider(resource=_build_resource(), metric_readers=[reader])


def setup_if_needed(plugin_cfg: _config.SigilPluginConfig) -> bool:
    """Install TracerProvider and MeterProvider when missing.

    Returns True when at least one provider is in place after the call (host
    already had one, or this function installed one). Returns False when
    auto-setup is disabled and no provider exists.

    Idempotent — safe to call repeatedly. Each provider is considered
    independently: the host can own one and let the plugin install the other.
    """
    global _INSTALLED_TRACER_PROVIDER, _INSTALLED_METER_PROVIDER, _SETUP_DONE

    if _SETUP_DONE:
        return _INSTALLED_TRACER_PROVIDER is not None or _INSTALLED_METER_PROVIDER is not None or _has_any_provider()

    if not plugin_cfg.otel_configured:
        # No OTel endpoint env → only set up if the host has its own provider.
        _SETUP_DONE = True
        return _has_any_provider()

    from opentelemetry import metrics, trace

    current_tracer = trace.get_tracer_provider()
    current_meter = metrics.get_meter_provider()

    needs_tracer = _is_proxy_tracer_provider(current_tracer)
    needs_meter = _is_proxy_meter_provider(current_meter)

    if not (needs_tracer or needs_meter):
        # Host owns both already.
        _SETUP_DONE = True
        return True

    if not plugin_cfg.otel_auto:
        if needs_tracer or needs_meter:
            logger.warning(
                "hermes-plugin-sigil: SIGIL_HERMES_OTEL_AUTO=false and no provider is configured "
                "for %s — telemetry is disabled.",
                "TracerProvider+MeterProvider" if (needs_tracer and needs_meter)
                else ("TracerProvider" if needs_tracer else "MeterProvider"),
            )
        _SETUP_DONE = True
        return not (needs_tracer and needs_meter)

    try:
        if needs_tracer:
            provider = _install_tracer_provider()
            trace.set_tracer_provider(provider)
            _INSTALLED_TRACER_PROVIDER = provider
            logger.debug("hermes-plugin-sigil: installed TracerProvider with OTLP HTTP exporter")
        if needs_meter:
            provider = _install_meter_provider()
            metrics.set_meter_provider(provider)
            _INSTALLED_METER_PROVIDER = provider
            logger.debug("hermes-plugin-sigil: installed MeterProvider with OTLP HTTP exporter")
    except Exception as exc:
        logger.warning("hermes-plugin-sigil: failed to set up OTel providers: %s", exc)
        _SETUP_DONE = True
        return False

    _SETUP_DONE = True
    return True


def _has_any_provider() -> bool:
    from opentelemetry import metrics, trace

    return not _is_proxy_tracer_provider(trace.get_tracer_provider()) or not _is_proxy_meter_provider(
        metrics.get_meter_provider()
    )


def force_flush() -> None:
    """Flush the providers we installed. Skip ones owned by the host."""
    if _INSTALLED_TRACER_PROVIDER is not None:
        try:
            _INSTALLED_TRACER_PROVIDER.force_flush()
        except Exception as exc:
            logger.warning("hermes-plugin-sigil: TracerProvider force_flush failed: %s", exc)
    if _INSTALLED_METER_PROVIDER is not None:
        try:
            _INSTALLED_METER_PROVIDER.force_flush()
        except Exception as exc:
            logger.warning("hermes-plugin-sigil: MeterProvider force_flush failed: %s", exc)


def _reset_for_tests() -> None:
    """Clear cached install state and shut down any providers we installed.

    Shutdown is needed even in tests — the providers' background export threads
    otherwise keep firing retries against localhost long after the test ended.
    """
    global _INSTALLED_TRACER_PROVIDER, _INSTALLED_METER_PROVIDER, _SETUP_DONE
    if _INSTALLED_TRACER_PROVIDER is not None:
        try:
            _INSTALLED_TRACER_PROVIDER.shutdown()
        except Exception:
            pass
    if _INSTALLED_METER_PROVIDER is not None:
        try:
            _INSTALLED_METER_PROVIDER.shutdown()
        except Exception:
            pass
    _INSTALLED_TRACER_PROVIDER = None
    _INSTALLED_METER_PROVIDER = None
    _SETUP_DONE = False
