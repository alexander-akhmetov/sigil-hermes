"""Tests for OTel TracerProvider + MeterProvider auto-setup."""
from __future__ import annotations

import pytest
from opentelemetry import metrics, trace
from opentelemetry.metrics._internal import _ProxyMeterProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider

from hermes_plugin_sigil import _client, _config, _otel


@pytest.fixture(autouse=True)
def reset_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset both module-level state and the global tracer/meter providers."""
    _client._reset_for_tests()
    # Force a fresh ProxyTracerProvider for isolation. Safe: opentelemetry
    # exposes the proxy as a no-op default that downstream code tolerates.
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None, raising=False)
    monkeypatch.setattr(
        trace,
        "_TRACER_PROVIDER_SET_ONCE",
        trace._TRACER_PROVIDER_SET_ONCE.__class__(),
        raising=False,
    )
    # Same for metrics
    monkeypatch.setattr(metrics._internal, "_METER_PROVIDER", None, raising=False)
    monkeypatch.setattr(
        metrics._internal,
        "_METER_PROVIDER_SET_ONCE",
        metrics._internal._METER_PROVIDER_SET_ONCE.__class__(),
        raising=False,
    )
    yield
    _client._reset_for_tests()


def _proxy_tracer_active() -> bool:
    return isinstance(trace.get_tracer_provider(), ProxyTracerProvider)


def _proxy_meter_active() -> bool:
    return isinstance(metrics.get_meter_provider(), _ProxyMeterProvider)


@pytest.fixture
def otel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical OTel env: endpoint + auth that the SDK helpers will compose."""
    monkeypatch.setenv("SIGIL_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost/otlp")
    monkeypatch.setenv("SIGIL_AUTH_TENANT_ID", "stack-1")
    monkeypatch.setenv("SIGIL_AUTH_TOKEN", "glc_otlp_secret")


def _make_cfg(*, otel_auto: bool = True, otel_configured: bool = True) -> _config.SigilPluginConfig:
    return _config.SigilPluginConfig(
        otel_auto=otel_auto,
        otel_configured=otel_configured,
    )


def test_auto_setup_installs_both_providers_when_proxies_are_global(otel_env) -> None:
    assert _proxy_tracer_active(), "test fixture should leave a ProxyTracerProvider in place"
    assert _proxy_meter_active(), "test fixture should leave a proxy MeterProvider in place"
    cfg = _make_cfg(otel_auto=True)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    assert isinstance(metrics.get_meter_provider(), MeterProvider)


def test_auto_setup_is_idempotent(otel_env) -> None:
    cfg = _make_cfg(otel_auto=True)
    _otel.setup_if_needed(cfg)
    first_tracer = trace.get_tracer_provider()
    first_meter = metrics.get_meter_provider()
    _otel.setup_if_needed(cfg)
    assert trace.get_tracer_provider() is first_tracer
    assert metrics.get_meter_provider() is first_meter


def test_auto_setup_skipped_when_user_has_both_providers(otel_env) -> None:
    custom_tracer = TracerProvider()
    custom_meter = MeterProvider()
    trace.set_tracer_provider(custom_tracer)
    metrics.set_meter_provider(custom_meter)
    cfg = _make_cfg(otel_auto=True)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert trace.get_tracer_provider() is custom_tracer
    assert metrics.get_meter_provider() is custom_meter


def test_auto_setup_installs_only_missing_provider(otel_env) -> None:
    """When host owns one provider, plugin installs only the other."""
    custom_tracer = TracerProvider()
    trace.set_tracer_provider(custom_tracer)
    # Meter is still a proxy
    cfg = _make_cfg(otel_auto=True)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert trace.get_tracer_provider() is custom_tracer  # untouched
    assert isinstance(metrics.get_meter_provider(), MeterProvider)  # plugin installed


def test_auto_setup_disabled_returns_false_with_proxies(otel_env) -> None:
    cfg = _make_cfg(otel_auto=False)
    ok = _otel.setup_if_needed(cfg)
    assert ok is False
    assert _proxy_tracer_active(), "no provider must be installed when auto is disabled"
    assert _proxy_meter_active(), "no meter provider must be installed when auto is disabled"


def test_auto_setup_disabled_uses_existing_user_providers(otel_env) -> None:
    custom_tracer = TracerProvider()
    custom_meter = MeterProvider()
    trace.set_tracer_provider(custom_tracer)
    metrics.set_meter_provider(custom_meter)
    cfg = _make_cfg(otel_auto=False)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert trace.get_tracer_provider() is custom_tracer
    assert metrics.get_meter_provider() is custom_meter


def test_no_otel_env_is_no_op_for_otel() -> None:
    """When no OTel endpoint env is set, the plugin doesn't install a provider."""
    cfg = _make_cfg(otel_configured=False)
    ok = _otel.setup_if_needed(cfg)
    assert ok is False
    assert _proxy_tracer_active()
    assert _proxy_meter_active()


def test_sigil_otel_endpoint_takes_precedence_over_otel_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """SIGIL_OTEL_EXPORTER_OTLP_ENDPOINT must win over OTEL_EXPORTER_OTLP_ENDPOINT."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://other:4318")
    monkeypatch.setenv("SIGIL_OTEL_EXPORTER_OTLP_ENDPOINT", "http://sigil:4318")

    captured: dict[str, object] = {}

    class CaptureExporter(OTLPSpanExporter):  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        CaptureExporter,
    )
    cfg = _make_cfg(otel_auto=True)
    _otel.setup_if_needed(cfg)
    assert captured.get("endpoint") == "http://sigil:4318/v1/traces"
    # Process env stays untouched — plugin must not unset OTEL_*.
    import os

    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://other:4318"
