# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for exception-driven upstream Polars GPU gating."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from cudf_polars.testing import inject_gpu_engine


class _Config:
    def __init__(self, variant: str = "in-memory") -> None:
        self.options = {"--inject-gpu-engine": variant}

    def getoption(self, name: str) -> str:
        return self.options[name]


def _items(*nodeids: str) -> list[pytest.Item]:
    return [cast("pytest.Item", SimpleNamespace(nodeid=nodeid)) for nodeid in nodeids]


def test_spmd_exceptions_include_in_memory_exceptions() -> None:
    config = cast("pytest.Config", _Config("spmd"))
    expected_overrides = set(inject_gpu_engine.SPMD_EXPECTED_FAILURES)
    assert set(inject_gpu_engine._fallbacks_for(config)) == (
        set(inject_gpu_engine.CPU_FALLBACKS) - expected_overrides
    ) | set(inject_gpu_engine.SPMD_CPU_FALLBACKS)


def test_exception_lists_reject_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    config = cast("pytest.Config", _Config())
    nodeid = next(iter(inject_gpu_engine.CPU_FALLBACKS))
    monkeypatch.setitem(inject_gpu_engine.EXPECTED_FAILURES, nodeid, "overlap")
    with pytest.raises(pytest.UsageError, match="both CPU-fallback"):
        inject_gpu_engine._validate_exception_lists(config, _items(nodeid))


def test_exception_lists_reject_stale_node() -> None:
    config = cast("pytest.Config", _Config())
    with pytest.raises(pytest.UsageError, match="Stale"):
        inject_gpu_engine._validate_exception_lists(
            config, _items("tests/test_other.py::test_x")
        )
