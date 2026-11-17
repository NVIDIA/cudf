# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import io

import numpy as np
import pandas as pd
import pytest

import cudf


def test_series_init_none():
    # test for creating empty series
    # 1: without initializing
    sr1 = cudf.Series()
    got = sr1.to_string()

    expect = sr1.to_pandas().to_string()
    assert got == expect

    # 2: Using `None` as an initializer
    sr2 = cudf.Series(None)
    got = sr2.to_string()

    expect = sr2.to_pandas().to_string()
    assert got == expect


@pytest.mark.parametrize(
    "kwargs",
    [
        {"na_rep": "NULL"},
        {"float_format": "{:.3f}".format},
        {"header": False},
        {"index": False},
        {"length": True},
        {"dtype": True},
        {"name": True},
        {"max_rows": 3},
        {"max_rows": 3, "min_rows": 2},
        {"length": True, "dtype": True, "name": True},
    ],
    ids=[
        "na_rep",
        "float_format",
        "header_false",
        "index_false",
        "length",
        "dtype",
        "name",
        "max_rows",
        "min_rows",
        "footer",
    ],
)
def test_series_to_string_passthrough(kwargs):
    # Each explicit pandas argument is forwarded unchanged, so cuDF's
    # rendered output matches pandas exactly.
    ps = pd.Series(
        [1.5, 2.25, np.nan, 4.0], index=["w", "x", "y", "z"], name="vals"
    )
    gs = cudf.from_pandas(ps)
    assert gs.to_string(**kwargs) == ps.to_string(**kwargs)


def test_series_to_string_buf_writes_and_returns_none():
    # When ``buf`` is given, to_string writes to it and returns None,
    # matching pandas.
    ps = pd.Series([1, 2, 3], name="vals")
    gs = cudf.from_pandas(ps)
    gbuf, pbuf = io.StringIO(), io.StringIO()
    assert gs.to_string(buf=gbuf) is None
    ps.to_string(buf=pbuf)
    assert gbuf.getvalue() == pbuf.getvalue()


def test_series_to_string_arguments_are_keyword_only():
    # cuDF mirrors the canonical pandas signature (and pandas >= 4.0):
    # everything after ``buf`` is keyword-only. Note pandas 3.0 still
    # accepts these positionally with a deprecation warning; cuDF does not
    # carry that shim forward.
    gs = cudf.Series([1, 2, 3])
    with pytest.raises(TypeError):
        gs.to_string(None, "NULL")


def test_series_to_string_signature_matches_pandas():
    # The whole point of the explicit signature is to mirror pandas; guard
    # against future drift in parameter names, order, kind, or defaults.
    def spec(func):
        return [
            (param.name, param.kind, param.default)
            for param in inspect.signature(func).parameters.values()
            if param.name != "self"
        ]

    assert spec(cudf.Series.to_string) == spec(pd.Series.to_string)
