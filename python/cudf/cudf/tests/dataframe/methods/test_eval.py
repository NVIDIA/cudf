# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


# Note that for now expressions do not automatically handle casting, so inputs
# need to be casted appropriately
@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.eval is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "expr, dtype",
    [
        ("a", int),
        ("+a", int),
        ("a + b", int),
        ("a == b", int),
        ("a / b", float),
        ("a * b", int),
        ("a > b", int),
        ("a >= b", int),
        ("a > b > c", int),
        ("a > b < c", int),
        ("a & b", int),
        ("a & b | c", int),
        ("sin(a)", float),
        ("exp(sin(abs(a)))", float),
        ("sqrt(floor(a))", float),
        ("ceil(arctanh(a))", float),
        ("(a + b) - (c * d)", int),
        ("~a", int),
        ("(a > b) and (c > d)", int),
        ("(a > b) or (c > d)", int),
        ("not (a > b)", int),
        ("a + 1", int),
        ("a + 1.0", float),
        ("-a + 1", int),
        ("+a + 1", int),
        ("e = a + 1", int),
        (
            """
            e = log(cos(a)) + 1.0
            f = abs(c) - exp(d)
            """,
            float,
        ),
        ("a_b_are_equal = (a == b)", int),
        ("a > b", str),
        ("a < '1'", str),
        ('a == "1"', str),
    ],
)
@pytest.mark.parametrize("nrows", [0, 10])
def test_dataframe_eval(nrows, expr, dtype):
    arr = np.ones(nrows)
    df_eval = cudf.DataFrame({"a": arr, "b": arr, "c": arr, "d": arr})
    df_eval = df_eval.astype(dtype)
    expect = df_eval.to_pandas().eval(expr)
    got = df_eval.eval(expr)
    # In the specific case where the evaluated expression is a unary function
    # of a single column with no nesting, pandas will retain the name. This
    # level of compatibility is out of scope for now.
    assert_eq(expect, got, check_names=False)

    # Test inplace
    if re.search("[^=><]=[^=]", expr) is not None:
        pdf_eval = df_eval.to_pandas().eval(expr)
        df_eval.eval(expr, inplace=True)
        assert_eq(pdf_eval, df_eval)


@pytest.mark.parametrize(
    "expr",
    [
        """
        e = a + b
        a == b
        """,
        "a_b_are_equal = (a == b) = c",
    ],
)
def test_dataframe_eval_errors(expr):
    df = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        df.eval(expr)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.eval is deprecated:pandas.errors.Pandas4Warning"
)
def test_dataframe_eval_misc():
    df = cudf.DataFrame({"a": [1, 2, 3, None, 5]})
    got = df.eval("isnull(a)")
    assert_eq(got, cudf.Series.isnull(df["a"]), check_names=False)

    df.eval("c = isnull(1)", inplace=True)
    assert_eq(df["c"], cudf.Series([False] * len(df), name="c"))
