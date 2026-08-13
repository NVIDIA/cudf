# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cudf_polars.unstable."""

from __future__ import annotations

import warnings

import pytest

import polars.exceptions

import cudf_polars
from cudf_polars.unstable import UnstableWarning, issue_unstable_warning, unstable


def test_issue_unstable_warning_silent_by_default(monkeypatch):
    monkeypatch.delenv("CUDF_POLARS_WARN_UNSTABLE", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        issue_unstable_warning("test message")  # must not raise


def test_issue_unstable_warning_emits_when_enabled(monkeypatch):
    monkeypatch.setenv("CUDF_POLARS_WARN_UNSTABLE", "1")
    with pytest.warns(UnstableWarning, match="test message"):
        issue_unstable_warning("test message")


def test_issue_unstable_warning_default_message(monkeypatch):
    monkeypatch.setenv("CUDF_POLARS_WARN_UNSTABLE", "1")
    with pytest.warns(
        UnstableWarning, match="this functionality is considered unstable"
    ):
        issue_unstable_warning()


def test_issue_unstable_warning_appends_suffix(monkeypatch):
    monkeypatch.setenv("CUDF_POLARS_WARN_UNSTABLE", "1")
    with pytest.warns(UnstableWarning, match="breaking change"):
        issue_unstable_warning("something")


def test_unstable_decorator_preserves_metadata():
    @unstable()
    def my_func(x: int) -> int:
        """Original docstring."""
        return x

    assert my_func.__name__ == "my_func"
    assert "Original docstring." in (my_func.__doc__ or "")


def test_unstable_decorator_appends_warning_to_docstring():
    @unstable()
    def my_func():
        """Original docstring."""

    assert "unstable" in (my_func.__doc__ or "").lower()
    assert "Warns" in (my_func.__doc__ or "")


def test_unstable_decorator_preserves_signature():
    import inspect

    @unstable()
    def my_func(a: int, b: str = "x") -> bool:
        return True

    sig = inspect.signature(my_func)
    assert list(sig.parameters) == ["a", "b"]


def test_unstable_decorator_silent_by_default(monkeypatch):
    monkeypatch.delenv("CUDF_POLARS_WARN_UNSTABLE", raising=False)

    @unstable()
    def my_func():
        return 42

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert my_func() == 42


def test_unstable_decorator_warns_when_enabled(monkeypatch):
    monkeypatch.setenv("CUDF_POLARS_WARN_UNSTABLE", "1")

    @unstable()
    def my_func():
        return 42

    with pytest.warns(UnstableWarning, match="my_func"):
        result = my_func()
    assert result == 42


def test_unstable_decorator_works_on_method(monkeypatch):
    monkeypatch.setenv("CUDF_POLARS_WARN_UNSTABLE", "1")

    class MyClass:
        @unstable()
        def my_method(self):
            return "ok"

    with pytest.warns(UnstableWarning, match="my_method"):
        assert MyClass().my_method() == "ok"


def test_unstable_warning_is_subclass_of_polars_unstable_warning():
    assert issubclass(UnstableWarning, polars.exceptions.UnstableWarning)


def test_unstable_warning_exported_from_top_level_package():
    assert cudf_polars.UnstableWarning is UnstableWarning
