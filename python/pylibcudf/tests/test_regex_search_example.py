# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pylibcudf regex-search example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from rmm.pylibrmm.stream import DEFAULT_STREAM

import pylibcudf as plc

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "regex_search.py"
)
SPEC = importlib.util.spec_from_file_location(
    "regex_search_example", EXAMPLE_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
regex_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = regex_search
SPEC.loader.exec_module(regex_search)


@pytest.mark.parametrize(
    "content,pattern,ignore_case,expected_lines,expected_matches",
    [
        pytest.param(
            "ERROR first message\nerror second message\nERROR third message\n",
            "ERROR",
            False,
            3,
            2,
            id="case-sensitive",
        ),
        pytest.param(
            "ERROR first message\nerror second message\nERROR third message\n",
            "error",
            True,
            3,
            3,
            id="ignore-case",
        ),
        pytest.param("", "ERROR", False, 0, 0, id="empty-file"),
        pytest.param("ERROR only\n", "ERROR", False, 1, 1, id="single-line"),
        pytest.param(
            "INFO first\nERROR final",
            "ERROR",
            False,
            2,
            1,
            id="no-final-newline",
        ),
    ],
)
def test_benchmark_file(
    tmp_path,
    content,
    pattern,
    ignore_case,
    expected_lines,
    expected_matches,
):
    """Benchmark regular and boundary-case file layouts accurately."""
    path = tmp_path / "log.txt"
    path.write_text(content, encoding="utf-8")

    result = regex_search.benchmark_file(
        str(path),
        pattern,
        gds=False,
        ignore_case=ignore_case,
        repeats=2,
        plc=plc,
        stream=DEFAULT_STREAM,
    )

    assert result.bytes == path.stat().st_size
    assert result.lines == expected_lines
    assert result.matches == expected_matches
    assert result.end_to_end_seconds > 0
    assert result.scan_seconds > 0


def test_configure_gds(monkeypatch):
    """Configure KvikIO compatibility mode for the requested I/O path."""
    monkeypatch.delenv("KVIKIO_COMPAT_MODE", raising=False)

    regex_search.configure_gds(False)
    assert regex_search.os.environ["KVIKIO_COMPAT_MODE"] == "ON"

    regex_search.configure_gds(True)
    assert regex_search.os.environ["KVIKIO_COMPAT_MODE"] == "OFF"


def test_native_gds_fallback_is_rejected(monkeypatch):
    """Reject GDS mode when native GDS is unavailable."""
    cufile_driver = type(
        "CuFileDriver", (), {"get": staticmethod(lambda name: False)}
    )
    kvikio = type("KvikIO", (), {"cufile_driver": cufile_driver})
    monkeypatch.setitem(sys.modules, "kvikio", kvikio)

    with pytest.raises(RuntimeError, match="native GDS is unavailable"):
        regex_search._require_native_gds()


def test_native_gds_requires_fallback_to_be_disabled(monkeypatch):
    """Reject GDS mode while cuFile compatibility fallback is allowed."""
    settings = {"is_gds_available": True, "allow_compat_mode": True}
    cufile_driver = type(
        "CuFileDriver", (), {"get": staticmethod(settings.__getitem__)}
    )
    kvikio = type("KvikIO", (), {"cufile_driver": cufile_driver})
    monkeypatch.setitem(sys.modules, "kvikio", kvikio)

    with pytest.raises(RuntimeError, match="allow_compat_mode=false"):
        regex_search._require_native_gds()
