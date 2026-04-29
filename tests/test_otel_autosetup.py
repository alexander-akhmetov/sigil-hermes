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


def _make_cfg(otel_auto: bool = True) -> _config.SigilPluginConfig:
    return _config.SigilPluginConfig(
        generations=None,
        otlp=_config.OTLPConfig(
            endpoint="http://localhost/otlp",
            instance_id="stack-1",
            token="glc_otlp_secret",
        ),
        otel_auto=otel_auto,
    )


def test_auto_setup_installs_both_providers_when_proxies_are_global() -> None:
    assert _proxy_tracer_active(), "test fixture should leave a ProxyTracerProvider in place"
    assert _proxy_meter_active(), "test fixture should leave a proxy MeterProvider in place"
    cfg = _make_cfg(otel_auto=True)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    assert isinstance(metrics.get_meter_provider(), MeterProvider)


def test_auto_setup_is_idempotent() -> None:
    cfg = _make_cfg(otel_auto=True)
    _otel.setup_if_needed(cfg)
    first_tracer = trace.get_tracer_provider()
    first_meter = metrics.get_meter_provider()
    _otel.setup_if_needed(cfg)
    assert trace.get_tracer_provider() is first_tracer
    assert metrics.get_meter_provider() is first_meter


def test_auto_setup_skipped_when_user_has_both_providers() -> None:
    custom_tracer = TracerProvider()
    custom_meter = MeterProvider()
    trace.set_tracer_provider(custom_tracer)
    metrics.set_meter_provider(custom_meter)
    cfg = _make_cfg(otel_auto=True)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert trace.get_tracer_provider() is custom_tracer
    assert metrics.get_meter_provider() is custom_meter


def test_auto_setup_installs_only_missing_provider() -> None:
    """When host owns one provider, plugin installs only the other."""
    custom_tracer = TracerProvider()
    trace.set_tracer_provider(custom_tracer)
    # Meter is still a proxy
    cfg = _make_cfg(otel_auto=True)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert trace.get_tracer_provider() is custom_tracer  # untouched
    assert isinstance(metrics.get_meter_provider(), MeterProvider)  # plugin installed


def test_auto_setup_disabled_returns_false_with_proxies() -> None:
    cfg = _make_cfg(otel_auto=False)
    ok = _otel.setup_if_needed(cfg)
    assert ok is False
    assert _proxy_tracer_active(), "no provider must be installed when auto is disabled"
    assert _proxy_meter_active(), "no meter provider must be installed when auto is disabled"


def test_auto_setup_disabled_uses_existing_user_providers() -> None:
    custom_tracer = TracerProvider()
    custom_meter = MeterProvider()
    trace.set_tracer_provider(custom_tracer)
    metrics.set_meter_provider(custom_meter)
    cfg = _make_cfg(otel_auto=False)
    ok = _otel.setup_if_needed(cfg)
    assert ok is True
    assert trace.get_tracer_provider() is custom_tracer
    assert metrics.get_meter_provider() is custom_meter


def test_no_otlp_config_is_no_op_for_otel() -> None:
    """When OTLP creds are missing, plugin doesn't install any provider."""
    cfg = _config.SigilPluginConfig(generations=None, otlp=None)
    ok = _otel.setup_if_needed(cfg)
    assert ok is False
    assert _proxy_tracer_active()
    assert _proxy_meter_active()
