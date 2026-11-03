# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("fill_value", [[888, 999]])
def test_dataframe_with_nulls_where_with_scalars(fill_value):
    pdf = pd.DataFrame(
        {
            "A": [-1, 2, -3, None, 5, 6, -7, 0],
            "B": [4, -2, 3, None, 7, 6, 8, 0],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.where(pdf % 3 == 0, fill_value)
    got = gdf.where(gdf % 3 == 0, fill_value)

    assert_eq(expect, got)


def test_dataframe_with_different_types(request):
    # Testing for int and float
    pdf = pd.DataFrame(
        {"A": [111, 22, 31, 410, 56], "B": [-10.12, 121.2, 45.7, 98.4, 87.6]}
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.where(pdf > 50, -pdf)
    got = gdf.where(gdf > 50, -gdf)

    assert_eq(expect, got)

    # Testing for string
    request.applymarker(
        pytest.mark.xfail(
            reason="https://github.com/pandas-dev/pandas/issues/63842"
        )
    )
    pdf = pd.DataFrame({"A": ["a", "bc", "cde", "fghi"]})
    gdf = cudf.from_pandas(pdf)
    pdf_mask = pd.DataFrame({"A": [True, False, True, False]})
    gdf_mask = cudf.from_pandas(pdf_mask)
    expect = pdf.where(pdf_mask, ["cudf"])
    got = gdf.where(gdf_mask, ["cudf"])
    assert_eq(expect, got)

    # Testing for categoriacal
    pdf = pd.DataFrame({"A": ["a", "b", "b", "c"]})
    pdf["A"] = pdf["A"].astype("category")
    gdf = cudf.from_pandas(pdf)
    expect = pdf.where(pdf_mask, "c")
    got = gdf.where(gdf_mask, ["c"])

    assert_eq(expect, got)


def test_dataframe_where_with_different_options():
    pdf = pd.DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]})
    gdf = cudf.from_pandas(pdf)

    # numpy array
    boolean_mask = np.array([[False, True], [True, False], [False, True]])

    expect = pdf.where(boolean_mask, -pdf)
    got = gdf.where(boolean_mask, -gdf)

    assert_eq(expect, got)

    # with single scalar
    expect = pdf.where(boolean_mask, 8)
    got = gdf.where(boolean_mask, 8)

    assert_eq(expect, got)

    # with multi scalar
    expect = pdf.where(boolean_mask, [8, 9])
    got = gdf.where(boolean_mask, [8, 9])

    assert_eq(expect, got)


def test_frame_series_where():
    gdf = cudf.DataFrame(
        {"a": [1.0, 2.0, None, 3.0, None], "b": [None, 10.0, 11.0, None, 23.0]}
    )
    pdf = gdf.to_pandas()
    expected = gdf.where(gdf.notna(), gdf.mean())
    actual = pdf.where(pdf.notna(), pdf.mean(), axis=1)
    assert_eq(expected, actual)


def test_frame_series_where_other():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [1, 1, 0]})
    pdf = gdf.to_pandas()

    expected = gdf.where(gdf["b"] == 1, cudf.NA)
    actual = pdf.where(pdf["b"] == 1, pd.NA)
    assert_eq(
        actual.fillna(-1).values,
        expected.fillna(-1).values,
        check_dtype=False,
    )

    expected = gdf.where(gdf["b"] == 1, 0)
    actual = pdf.where(pdf["b"] == 1, 0)
    assert_eq(expected, actual)


@pytest.mark.parametrize("method", ["where", "mask"])
def test_frame_series_condition_nondefault_index(method):
    pdf = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["i0", "i1"])
    gdf = cudf.from_pandas(pdf)
    pcond = pd.Series([True, False], index=pdf.index)
    gcond = cudf.from_pandas(pcond)

    expected = getattr(pdf, method)(pcond, 100)
    actual = getattr(gdf, method)(gcond, 100)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,condition,other,error",
    [
        (pd.Series(range(5)), pd.Series(range(5)) > 0, None, None),
        (pd.Series(range(5)), pd.Series(range(5)) > 1, None, None),
        (pd.Series(range(5)), pd.Series(range(5)) > 1, 10, None),
        (
            pd.Series(range(5)),
            pd.Series(range(5)) > 1,
            pd.Series(range(5, 10)),
            None,
        ),
        (
            pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"]),
            (
                pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
                % 3
            )
            == 0,
            -pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"]),
            None,
        ),
        (
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}),
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}) == 4,
            None,
            None,
        ),
        (
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}),
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}) != 4,
            None,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [True, True, True],
            None,
            ValueError,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [True, True, True, False],
            None,
            ValueError,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [[True, True, True, False], [True, True, True, False]],
            None,
            ValueError,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [[True, True], [False, True], [True, False], [False, True]],
            None,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            cp.array(
                [[True, True], [False, True], [True, False], [False, True]]
            ),
            None,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            cp.array(
                [[True, True], [False, True], [True, False], [False, True]]
            ),
            17,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [[True, True], [False, True], [True, False], [False, True]],
            17,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [
                [True, True, False, True],
                [True, True, False, True],
                [True, True, False, True],
                [True, True, False, True],
            ],
            None,
            ValueError,
        ),
        (
            pd.Series([1, 2, np.nan]),
            pd.Series([1, 2, np.nan]) == 4,
            None,
            None,
        ),
        (
            pd.Series([1, 2, np.nan]),
            pd.Series([1, 2, np.nan]) != 4,
            None,
            None,
        ),
        (
            pd.Series([4, np.nan, 6]),
            pd.Series([4, np.nan, 6]) == 4,
            None,
            None,
        ),
        (
            pd.Series([4, np.nan, 6]),
            pd.Series([4, np.nan, 6]) != 4,
            None,
            None,
        ),
        (
            pd.Series([4, np.nan, 6], dtype="category"),
            pd.Series([4, np.nan, 6], dtype="category") != 4,
            None,
            None,
        ),
        (
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category"),
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category") == "b",
            None,
            None,
        ),
        (
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category"),
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category") == "b",
            "s",
            None,
        ),
        (
            pd.Series([1, 2, 3, 2, 5]),
            pd.Series([1, 2, 3, 2, 5]) == 2,
            pd.DataFrame(
                {
                    "a": pd.Series([1, 2, 3, 2, 5]),
                    "b": pd.Series([1, 2, 3, 2, 5]),
                }
            ),
            NotImplementedError,
        ),
    ],
)
def test_df_sr_mask_where(data, condition, other, error, inplace):
    ps_where = data
    gs_where = cudf.from_pandas(data)

    ps_mask = ps_where.copy(deep=True)
    gs_mask = gs_where.copy(deep=True)

    if hasattr(condition, "__cuda_array_interface__"):
        if type(condition).__module__.split(".")[0] == "cupy":
            ps_condition = cp.asnumpy(condition)
        else:
            ps_condition = np.array(condition).astype("bool")
    else:
        ps_condition = condition

    if type(condition).__module__.split(".")[0] == "pandas":
        gs_condition = cudf.from_pandas(condition)
    else:
        gs_condition = condition

    ps_other = other
    if type(other).__module__.split(".")[0] == "pandas":
        gs_other = cudf.from_pandas(other)
    else:
        gs_other = other

    if error is None:
        expect_where = ps_where.where(
            ps_condition, other=ps_other, inplace=inplace
        )
        got_where = gs_where.where(
            gs_condition, other=gs_other, inplace=inplace
        )

        expect_mask = ps_mask.mask(
            ps_condition, other=ps_other, inplace=inplace
        )
        got_mask = gs_mask.mask(gs_condition, other=gs_other, inplace=inplace)

        if inplace:
            expect_where = ps_where
            got_where = gs_where

            expect_mask = ps_mask
            got_mask = gs_mask

        if isinstance(expect_where, pd.Series) and isinstance(
            expect_where.dtype, pd.CategoricalDtype
        ):
            np.testing.assert_array_equal(
                expect_where.cat.codes,
                got_where.cat.codes.astype(expect_where.cat.codes.dtype)
                .fillna(-1)
                .to_numpy(),
            )
            assert_eq(expect_where.cat.categories, got_where.cat.categories)

            np.testing.assert_array_equal(
                expect_mask.cat.codes,
                got_mask.cat.codes.astype(expect_mask.cat.codes.dtype)
                .fillna(-1)
                .to_numpy(),
            )
            assert_eq(expect_mask.cat.categories, got_mask.cat.categories)
        else:
            assert_eq(
                expect_where.fillna(-1),
                got_where.fillna(-1),
                check_dtype=False,
            )
            assert_eq(
                expect_mask.fillna(-1), got_mask.fillna(-1), check_dtype=False
            )
    else:
        assert_exceptions_equal(
            lfunc=ps_where.where,
            rfunc=gs_where.where,
            lfunc_args_and_kwargs=(
                [ps_condition],
                {"other": ps_other, "inplace": inplace},
            ),
            rfunc_args_and_kwargs=(
                [gs_condition],
                {"other": gs_other, "inplace": inplace},
            ),
        )

        assert_exceptions_equal(
            lfunc=ps_mask.mask,
            rfunc=gs_mask.mask,
            lfunc_args_and_kwargs=(
                [ps_condition],
                {"other": ps_other, "inplace": inplace},
            ),
            rfunc_args_and_kwargs=(
                [gs_condition],
                {"other": gs_other, "inplace": inplace},
            ),
        )


@pytest.mark.parametrize(
    "data,condition,other,has_cat",
    [
        (
            pd.DataFrame(
                {
                    "a": pd.Series(["a", "a", "b", "c", "a", "d", "d", "a"]),
                    "b": pd.Series(["o", "p", "q", "e", "p", "p", "a", "a"]),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(["a", "a", "b", "c", "a", "d", "d", "a"]),
                    "b": pd.Series(["o", "p", "q", "e", "p", "p", "a", "a"]),
                }
            )
            != "a",
            None,
            None,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            != "a",
            None,
            True,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            == "a",
            None,
            True,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            != "a",
            "a",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            == "a",
            "a",
            True,
        ),
    ],
)
def test_df_string_cat_types_mask_where(data, condition, other, has_cat):
    ps = data
    gs = cudf.from_pandas(data)

    ps_condition = condition
    if type(condition).__module__.split(".")[0] == "pandas":
        gs_condition = cudf.from_pandas(condition)
    else:
        gs_condition = condition

    ps_other = other
    if type(other).__module__.split(".")[0] == "pandas":
        gs_other = cudf.from_pandas(other)
    else:
        gs_other = other

    expect_where = ps.where(ps_condition, other=ps_other)
    got_where = gs.where(gs_condition, other=gs_other)

    expect_mask = ps.mask(ps_condition, other=ps_other)
    got_mask = gs.mask(gs_condition, other=gs_other)

    if has_cat is None:
        assert_eq(
            expect_where.fillna(-1).astype("str"),
            got_where.fillna(-1),
            check_dtype=False,
        )
        assert_eq(
            expect_mask.fillna(-1).astype("str"),
            got_mask.fillna(-1),
            check_dtype=False,
        )
    else:
        assert_eq(expect_where, got_where, check_dtype=False)
        assert_eq(expect_mask, got_mask, check_dtype=False)
