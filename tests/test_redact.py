"""Tests for the structural payload redactor."""
from __future__ import annotations

from hermes_plugin_sigil import _redact


def test_truncate_long_string_uses_caller_max_chars() -> None:
    long = "a" * 20000
    result = _redact.safe_value(long, max_chars=12000)
    assert result.startswith("a" * 12000)
    assert "truncated" in result


def test_short_max_chars_truncates_aggressively() -> None:
    result = _redact.safe_value("a" * 50, max_chars=10)
    assert result.startswith("a" * 10)
    assert "truncated" in result


def test_max_chars_kwarg_is_required() -> None:
    # The signature mandates max_chars; calling without it must error.
    try:
        _redact.safe_value("hi")  # type: ignore[call-arg]
    except TypeError:
        return
    raise AssertionError("safe_value should require max_chars")


def test_depth_limit_caps_at_4() -> None:
    # depth > 4 returns sentinel — top-level dict is depth 0, so the value at l5
    # is encountered at depth 5 and replaced.
    nested = {"l1": {"l2": {"l3": {"l4": {"l5": "deep"}}}}}
    out = _redact.safe_value(nested, max_chars=12000)
    cur = out
    for level in ("l1", "l2", "l3", "l4", "l5"):
        cur = cur[level]
    assert cur == "<max-depth>"


def test_within_depth_limit_preserves_values() -> None:
    nested = {"l1": {"l2": {"l3": "ok"}}}
    out = _redact.safe_value(nested, max_chars=12000)
    assert out == {"l1": {"l2": {"l3": "ok"}}}


def test_dict_entry_cap_is_50() -> None:
    big_dict = {f"k{i}": i for i in range(200)}
    out = _redact.safe_value(big_dict, max_chars=12000)
    assert len(out) == 50


def test_list_entry_cap_is_50() -> None:
    big_list = list(range(200))
    out = _redact.safe_value(big_list, max_chars=12000)
    assert len(out) == 50
    assert out[0] == 0
    assert out[-1] == 49


def test_scalars_pass_through_unchanged() -> None:
    assert _redact.safe_value(None, max_chars=12000) is None
    assert _redact.safe_value(42, max_chars=12000) == 42
    assert _redact.safe_value(3.14, max_chars=12000) == 3.14
    assert _redact.safe_value(True, max_chars=12000) is True


def test_bytes_become_descriptor() -> None:
    out = _redact.safe_value(b"hello", max_chars=12000)
    assert out == {"type": "bytes", "len": 5}


def test_parse_json_strings_when_requested() -> None:
    s = '{"a": 1, "b": [1, 2, 3]}'
    out = _redact.safe_value(s, max_chars=12000, parse_json_strings=True)
    assert out == {"a": 1, "b": [1, 2, 3]}


def test_unparseable_json_string_returned_as_string() -> None:
    s = "not json {{"
    out = _redact.safe_value(s, max_chars=12000, parse_json_strings=True)
    assert out == "not json {{"


def test_object_without_dict_attribute_falls_back_to_repr() -> None:
    class Opaque:
        __slots__ = ()

        def __repr__(self) -> str:
            return "<opaque>"

    out = _redact.safe_value(Opaque(), max_chars=12000)
    assert out == "<opaque>"
