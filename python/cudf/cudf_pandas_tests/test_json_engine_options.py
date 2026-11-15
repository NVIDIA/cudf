# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.fixture
def json_lines_path(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"a":1,"b":2}\n{"a":3,"b":4}\n', encoding="utf-8")
    return path


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="New pandas 3.1 engine validation"
)
@pytest.mark.parametrize(
    "options",
    [
        {"typ": "series"},
        {"convert_axes": True},
        {"convert_axes": False},
        {"convert_dates": False},
        {"convert_dates": ["a"]},
        {"keep_default_dates": False},
        {"precise_float": True},
        {"date_unit": "s"},
        {"dtype": True},
        {"dtype": False},
        {"dtype": "float64"},
        {"dtype": {"a": "int64[pyarrow]", "b": "float64"}},
        {"encoding": "latin-1"},
        {"encoding_errors": "ignore"},
        {"nrows": 1},
    ],
)
def test_pyarrow_json_engine_option_validation(json_lines_path, options):
    with pytest.raises((ValueError, NotImplementedError)) as expected:
        pd.read_json(json_lines_path, lines=True, engine="pyarrow", **options)
    with pytest.raises(type(expected.value), match="") as actual:
        xpd.read_json(json_lines_path, lines=True, engine="pyarrow", **options)
    assert str(actual.value) == str(expected.value)


def test_pyarrow_json_engine_valid_read(json_lines_path):
    expected = pd.read_json(json_lines_path, lines=True, engine="pyarrow")
    actual = xpd.read_json(json_lines_path, lines=True, engine="pyarrow")
    assert_eq(expected, actual._fsproxy_slow)


def test_default_json_engine_stays_on_gpu(json_lines_path, monkeypatch):
    with monkeypatch.context() as context:
        context.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "1")
        actual = xpd.read_json(json_lines_path, lines=True)
    expected = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert_eq(expected, actual._fsproxy_slow)
