# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import cudf
from cudf.testing import assert_eq


def test_from_pandas_intervaldtype():
    dtype = pd.IntervalDtype("int64", closed="left")
    result = cudf.from_pandas(dtype)
    expected = cudf.IntervalDtype("int64", closed="left")
    assert_eq(result, expected)


def test_intervaldtype_eq_string_with_attributes():
    dtype = cudf.IntervalDtype("int64", closed="left")
    assert dtype == "interval"
    assert dtype == "interval[int64, left]"


def test_empty_intervaldtype():
    # "older pandas" supported closed=None, cudf chooses not to support that
    pd_id = pd.IntervalDtype(closed="right")
    cudf_id = cudf.IntervalDtype()

    assert str(pd_id) == str(cudf_id)
    assert pd_id.subtype == cudf_id.subtype


def test_interval_dtype_pyarrow_round_trip(
    signed_integer_types_as_str, interval_closed
):
    pa_array = ArrowIntervalType(signed_integer_types_as_str, interval_closed)
    expect = pa_array
    got = cudf.IntervalDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_interval_dtype_from_pandas(
    signed_integer_types_as_str, interval_closed
):
    expect = cudf.IntervalDtype(
        signed_integer_types_as_str, closed=interval_closed
    )
    pd_type = pd.IntervalDtype(
        signed_integer_types_as_str, closed=interval_closed
    )
    got = cudf.IntervalDtype(pd_type.subtype, closed=pd_type.closed)
    assert expect == got


# GH#14273: IntervalDtype now supports ``closed=None`` and pandas-compatible
# string construction. The tests below assert parity with pandas.
@pytest.mark.parametrize(
    "arg",
    [
        "interval",
        "Interval",
        "interval[int64]",
        "interval[int64, left]",
        "interval[int64, right]",
        "interval[int64, both]",
        "interval[int64, neither]",
        "interval[datetime64[ns]]",
        "interval[datetime64[ns], both]",
        "int64",
        np.int64,
    ],
)
def test_intervaldtype_construction_matches_pandas(arg):
    got = cudf.IntervalDtype(arg)
    expect = pd.IntervalDtype(arg)
    assert got.subtype == expect.subtype
    assert got.closed == expect.closed
    assert str(got) == str(expect)


def test_intervaldtype_default_closed_is_none():
    # closed defaults to None (matching pandas), not "right".
    assert cudf.IntervalDtype("int64").closed is None
    assert cudf.IntervalDtype().closed is None
    assert cudf.IntervalDtype().subtype is None


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_intervaldtype_string_closed_reconciliation(closed):
    # A closed embedded in the string fills in a None ``closed`` keyword.
    from_string = cudf.IntervalDtype(f"interval[int64, {closed}]")
    explicit = cudf.IntervalDtype("int64", closed=closed)
    assert from_string == explicit
    # Passing a matching ``closed`` keyword is allowed.
    assert cudf.IntervalDtype(f"interval[int64, {closed}]", closed) == explicit


def test_intervaldtype_string_closed_conflict_raises():
    with pytest.raises(ValueError, match="does not match"):
        cudf.IntervalDtype("interval[int64, right]", closed="left")


@pytest.mark.parametrize("bad", ["interval[object]", "interval[category]"])
def test_intervaldtype_unsupported_subtype_raises(bad):
    with pytest.raises(TypeError):
        cudf.IntervalDtype(bad)


@pytest.mark.parametrize(
    "obj",
    [
        cudf.IntervalDtype("int64", closed="left"),
        pd.IntervalDtype("int64", closed="left"),
    ],
)
def test_intervaldtype_unwraps_interval_dtype(obj):
    result = cudf.IntervalDtype(obj)
    assert result.subtype == cudf.dtype("int64")
    assert result.closed == "left"


def test_intervaldtype_unwrap_closed_conflict_raises():
    with pytest.raises(ValueError, match="do not match"):
        cudf.IntervalDtype(
            cudf.IntervalDtype("int64", closed="left"), closed="right"
        )


@pytest.mark.parametrize(
    "string", ["interval", "interval[int64]", "interval[int64, left]"]
)
def test_intervaldtype_construct_from_string(string):
    assert cudf.IntervalDtype.construct_from_string(
        string
    ) == cudf.IntervalDtype(string)


@pytest.mark.parametrize("bad", ["int64", "foo", 123])
def test_intervaldtype_construct_from_string_invalid(bad):
    with pytest.raises(TypeError):
        cudf.IntervalDtype.construct_from_string(bad)
