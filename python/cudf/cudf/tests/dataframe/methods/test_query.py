# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import datetime
import inspect

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame
from cudf.testing import assert_eq
from cudf.utils import queryutils


@pytest.mark.parametrize(
    "text,expect_args",
    [
        ("a > @b", ("a", "__CUDF_ENVREF__b")),
        ("(a + b) <= @c", ("a", "b", "__CUDF_ENVREF__c")),
        ("a > b if a > 0 else b > a", ("a", "b")),
    ],
)
def test_query_parser(text, expect_args):
    info = queryutils.query_parser(text)
    fn = queryutils.query_builder(info, "myfoo")
    assert callable(fn)
    argspec = inspect.getfullargspec(fn)
    assert tuple(argspec.args) == tuple(expect_args)


@pytest.mark.parametrize(
    "fn",
    [
        (lambda a, b: a < b, "a < b"),
        (lambda a, b: a * 2 >= b, "a * 2 >= b"),
        (lambda a, b: 2 * (a + b) > (a + b) / 2, "2 * (a + b) > (a + b) / 2"),
    ],
)
@pytest.mark.parametrize("nulls", [True, False])
def test_query(fn, nulls):
    n = 5
    _expect_fn, query_expr = fn
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": np.arange(n),
            "b": rng.random(n) * n,
        }
    )
    if nulls:
        pdf.loc[::2, "a"] = None
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.query(query_expr), gdf.query(query_expr))


@pytest.mark.parametrize(
    "fn",
    [
        (lambda a, b, c, d: a * c > b + d, "a * @c > b + @d"),
        (
            lambda a, b, c, d: ((a / c) < d) | ((b**c) > d),
            "((a / @c) < @d) | ((b ** @c) > @d)",
        ),
    ],
)
def test_query_ref_env(fn):
    n = 5
    expect_fn, query_expr = fn
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    df["a"] = aa = np.arange(n)
    df["b"] = bb = rng.random(n) * n
    c = 2.3
    d = 1.2
    # udt
    expect_mask = expect_fn(aa, bb, c, d)
    df2 = df.query(query_expr)
    # check
    assert len(df2) == np.count_nonzero(expect_mask)
    np.testing.assert_array_almost_equal(df2["a"].to_numpy(), aa[expect_mask])
    np.testing.assert_array_almost_equal(df2["b"].to_numpy(), bb[expect_mask])


def test_query_env_changing():
    df = DataFrame()
    df["a"] = aa = np.arange(100)
    expr = "a < @c"
    # first attempt
    c = 10
    got = df.query(expr)
    np.testing.assert_array_equal(aa[aa < c], got["a"].to_numpy())
    # change env
    c = 50
    got = df.query(expr)
    np.testing.assert_array_equal(aa[aa < c], got["a"].to_numpy())


def test_query_local_dict():
    df = DataFrame()
    df["a"] = aa = np.arange(100)
    expr = "a < @val"

    got = df.query(expr, local_dict={"val": 10})
    np.testing.assert_array_equal(aa[aa < 10], got["a"].to_numpy())


def test_query_local_dict_datetime():
    df = DataFrame(
        {
            "datetimes": np.array(
                ["2018-10-07", "2018-10-08"], dtype="datetime64"
            )
        }
    )
    search_date = datetime.datetime.strptime("2018-10-08", "%Y-%m-%d")
    expr = "datetimes==@search_date"

    got = df.query(expr, local_dict={"search_date": search_date})
    np.testing.assert_array_equal(
        np.datetime64("2018-10-08"), got["datetimes"].to_numpy()
    )


def test_query_global_dict():
    df = DataFrame()
    df["a"] = aa = np.arange(100)
    expr = "a < @foo"

    global_scope = {"foo": 42}
    got = df.query(expr, global_dict=global_scope)

    np.testing.assert_array_equal(aa[aa < 42], got["a"].to_numpy())

    global_scope = {"foo": 7}
    got = df.query(expr, global_dict=global_scope)

    np.testing.assert_array_equal(aa[aa < 7], got["a"].to_numpy())


def test_query_splitted_combine():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=10), "y": rng.normal(size=10)}
    )
    gdf = DataFrame(df)

    # Split the GDF
    s1 = gdf[:5]
    s2 = gdf[5:]

    # Do the query
    expr = "x > 2"
    q1 = s1.query(expr)
    q2 = s2.query(expr)
    # Combine
    got = cudf.concat([q1, q2]).to_pandas()

    # Should equal to just querying the original GDF
    expect = gdf.query(expr).to_pandas()
    assert_eq(got, expect, check_index_type=True)


def test_query_empty_frames():
    empty_pdf = pd.DataFrame({"a": [], "b": []})
    empty_gdf = DataFrame(empty_pdf)
    # Do the query
    expr = "a > 2"
    got = empty_gdf.query(expr).to_pandas()
    expect = empty_pdf.query(expr)

    # assert equal results
    assert_eq(got, expect)


@pytest.mark.parametrize("index", ["a", ["a", "b"]])
@pytest.mark.parametrize(
    "query",
    [
        "a < @a_val",
        "a < @a_val and b > @b_val",
        "(a < @a_val and b >@b_val) or c >@c_val",
    ],
)
def test_query_with_index_name(index, query):
    a_val = 4  # noqa: F841
    b_val = 3  # noqa: F841
    c_val = 15  # noqa: F841
    pdf = pd.DataFrame(
        {
            "a": [1, None, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [12, 15, 17, 19, 27],
        }
    )
    pdf.set_index(index)

    gdf = DataFrame(pdf)

    out = gdf.query(query)
    expect = pdf.query(query)

    assert_eq(out, expect)


@pytest.mark.parametrize(
    "query",
    [
        "index < @a_val",
        "index < @a_val and b > @b_val",
        "(index < @a_val and b >@b_val) or c >@c_val",
    ],
)
def test_query_with_index_keyword(query):
    a_val = 4  # noqa: F841
    b_val = 3  # noqa: F841
    c_val = 15  # noqa: F841
    pdf = pd.DataFrame(
        {
            "a": [1, None, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [12, 15, 17, 19, 27],
        }
    )
    pdf.set_index("a")

    gdf = DataFrame(pdf)

    out = gdf.query(query)
    expect = pdf.query(query)

    assert_eq(out, expect)


def test_query_unsupported_dtypes():
    query = "data == 'a'"
    gdf = cudf.DataFrame({"data": ["a", "b", "c"]})

    # make sure the query works in pandas
    pdf = gdf.to_pandas()
    pdf_result = pdf.query(query)

    expect = pd.DataFrame({"data": ["a"]})
    assert_eq(expect, pdf_result)

    # but fails in cuDF
    with pytest.raises(TypeError):
        gdf.query(query)


@pytest.mark.parametrize(
    "query",
    [
        "a == 3",
        pytest.param(
            "a != 3",
            marks=pytest.mark.xfail(reason="incompatible with pandas"),
        ),
        "a < 3",
        "a <= 3",
        "a < 3",
        "a >= 3",
    ],
)
def test_query_mask(nan_as_null, query):
    data = {"a": [0, 1.0, 2.0, None, 3, np.nan, None, 4]}
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data, nan_as_null=nan_as_null)

    pdf_q_res = pdf.query(query)
    gdf_q_res = gdf.query(query)

    assert_eq(pdf_q_res, gdf_q_res)
