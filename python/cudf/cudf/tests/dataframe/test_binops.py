# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "expected",
    [
        pd.RangeIndex(1, 2, name="a"),
        pd.Index([1], dtype=np.int8, name="a"),
        pd.MultiIndex.from_arrays([[1]], names=["a"]),
    ],
)
@pytest.mark.parametrize("binop", [lambda df: df == df, lambda df: df - 1])
def test_dataframe_binop_preserves_column_metadata(expected, binop):
    df = cudf.DataFrame([1], columns=expected)
    result = binop(df).columns
    pd.testing.assert_index_equal(result, expected, exact=True)


def test_dataframe_series_dot():
    pser = pd.Series(range(2))
    gser = cudf.from_pandas(pser)

    expected = pser @ pser
    actual = gser @ gser

    assert_eq(expected, actual)
    # 1-D dot/matmul on Series produces a 0-D scalar; cudf must return a
    # numpy scalar (e.g. ``np.int64``) to match pandas, not a Python int.
    assert type(actual) is type(expected)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"))
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pser], {}),
        rfunc_args_and_kwargs=([gser], {}),
    )

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pdf], {}),
        rfunc_args_and_kwargs=([gdf], {}),
    )

    pser = pd.Series(range(2), index=["a", "k"])
    gser = cudf.from_pandas(pser)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"), index=["a", "k"])
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    actual = gdf @ [2, 3]
    expected = pdf @ [2, 3]

    assert_eq(expected, actual)

    actual = pser @ [12, 13]
    expected = gser @ [12, 13]

    assert_eq(expected, actual)


def test_dataframe_binop_with_datetime_index():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(
        rng.random(2),
        index=pd.Index(
            [
                "2000-01-04",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        ),
    )
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


def test_dataframe_binop_and_where():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(rng.random(size=(2, 2)), columns=pd.Index([True, False]))
    gdf = cudf.from_pandas(df)

    expected = df > 1
    got = gdf > 1

    assert_eq(expected, got)

    expected = df[df > 1]
    got = gdf[gdf > 1]

    assert_eq(expected, got)


def test_dataframe_binop_with_mixed_string_types():
    rng = np.random.default_rng(seed=0)
    df1 = pd.DataFrame(rng.random(size=(3, 3)), columns=pd.Index([0, 1, 2]))
    df2 = pd.DataFrame(
        rng.random(size=(6, 6)),
        columns=pd.Index([0, 1, 2, "VhDoHxRaqt", "X0NNHBIPfA", "5FbhPtS0D1"]),
    )
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    expected = df2 + df1
    got = gdf2 + gdf1

    assert_eq(expected, got)


def test_dataframe_binop_with_mixed_date_types():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(rng.random(size=3), index=[0, 1, 2])
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"a": [10, 11, 12]}, index=["a", "b", "z"]),
        pd.DataFrame({"z": ["a"]}),
        pd.DataFrame({"a": [], "b": []}),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": ["a", "a", "c", "z", "A"], "z": [1, 2, 3, 4, 5]}),
    ],
)
def test_dataframe_error_equality(df1, df2, comparison_op):
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    assert_exceptions_equal(
        comparison_op, comparison_op, ([df1, df2],), ([gdf1, gdf2],)
    )


@pytest.mark.parametrize(
    "other",
    [
        1.0,
        pd.Series([1.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0], index=["x"]),
        pd.Series([1.0, 2.0], index=["x", "y"]),
        pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
    ],
)
def test_arithmetic_binops_df(arithmetic_op, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = arithmetic_op(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=arithmetic_op,
            rfunc=arithmetic_op,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = arithmetic_op(gdf, other)
        assert_eq(d, g)


@pytest.mark.parametrize(
    "other",
    [
        1.0,
        pd.Series([1.0, 2.0], index=["x", "y"]),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
    ],
)
def test_comparison_binops_df(comparison_op, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = comparison_op(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=comparison_op,
            rfunc=comparison_op,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = comparison_op(gdf, other)
        assert_eq(d, g)


@pytest.mark.parametrize(
    "other",
    [
        pd.Series([1.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0], index=["x"]),
        pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
    ],
)
def test_comparison_binops_df_reindexing(request, comparison_op, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = comparison_op(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=comparison_op,
            rfunc=comparison_op,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        request.applymarker(
            pytest.mark.xfail(
                condition=pdf.columns.difference(other.index).size > 0,
                reason="""
                Currently we will not match pandas for equality/inequality
                operators when there are columns that exist in a Series but not
                the DataFrame because pandas returns True/False values whereas
                we return NA. However, this reindexing is deprecated in pandas
                so we opt not to add support. This test should start passing
                once pandas removes the deprecated behavior in 2.0.  When that
                happens, this test can be merged with the two tests above into
                a single test with common parameters.
                """,
            )
        )

        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = comparison_op(gdf, other)
        assert_eq(d, g)


def test_binops_df_invalid():
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    with pytest.raises(TypeError):
        gdf + np.array([1, 2])


def test_bitwise_binops_df(bitwise_op):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    d = bitwise_op(pdf, pdf + 1)
    g = bitwise_op(gdf, gdf + 1)
    assert_eq(d, g)


def test_binops_series(binary_op):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf = pdf + 1.0
    gdf = gdf + 1.0
    d = binary_op(pdf.x, pdf.y)
    g = binary_op(gdf.x, gdf.y)
    assert_eq(d, g)


def test_bitwise_binops_series(bitwise_op):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    d = bitwise_op(pdf.x, pdf.y + 1)
    g = bitwise_op(gdf.x, gdf.y + 1)
    assert_eq(d, g)


@pytest.mark.parametrize("unaryop", [operator.neg, operator.inv, operator.abs])
@pytest.mark.parametrize(
    "col_name,assign_col_name", [(None, False), (None, True), ("abc", True)]
)
def test_unaryops_df(unaryop, col_name, assign_col_name):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    if assign_col_name:
        pdf.columns.name = col_name
    gdf = cudf.from_pandas(pdf)
    d = unaryop(pdf - 5)
    g = unaryop(gdf - 5)
    assert_eq(d, g)


def test_df_abs():
    rng = np.random.default_rng(seed=0)
    disturbance = pd.Series(rng.random(10))
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    pdf = pdf - 5 + disturbance
    d = np.abs(pdf)
    g = cudf.from_pandas(pdf).abs()
    assert_eq(d, g)


def test_scale_df():
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    got = (gdf - 5).scale()
    expect = cudf.DataFrame(
        {"x": np.linspace(0.0, 1.0, 10), "y": np.linspace(0.0, 1.0, 10)}
    )
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([4, 2, 3]),
        pd.Series([4, 2, 3], index=["a", "b", "c"]),
        pd.Series([4, 2, 3], index=["a", "b", "d"]),
        pd.Series([4, 2], index=["a", "b"]),
        pd.Series([4, 2, 3]),
        pd.Series([4, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"]),
    ],
)
@pytest.mark.parametrize("colnames", [["a", "b", "c"], [0, 1, 2]])
def test_df_sr_binop(psr, colnames, binary_op):
    data = [[3.0, 2.0, 5.0], [3.0, None, 5.0], [6.0, 7.0, np.nan]]
    data = dict(zip(colnames, data, strict=True))

    gsr = cudf.Series(psr).astype("Float64")

    gdf = cudf.DataFrame(data).astype("Float64")
    pdf = gdf.to_pandas()

    psr = gsr.to_pandas()

    try:
        expect = binary_op(pdf, psr)
    except ValueError:
        with pytest.raises(ValueError):
            binary_op(gdf, gsr)
        with pytest.raises(ValueError):
            binary_op(psr, pdf)
        with pytest.raises(ValueError):
            binary_op(gsr, gdf)
    else:
        got = binary_op(gdf, gsr)
        assert_eq(expect, got)

        expect = binary_op(psr, pdf)
        got = binary_op(gsr, gdf)
        assert_eq(expect, got)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        # comparison ops will temporarily XFAIL
        # see PR  https://github.com/rapidsai/cudf/pull/7491
        pytest.param(operator.eq, marks=pytest.mark.xfail),
        pytest.param(operator.lt, marks=pytest.mark.xfail),
        pytest.param(operator.le, marks=pytest.mark.xfail),
        pytest.param(operator.gt, marks=pytest.mark.xfail),
        pytest.param(operator.ge, marks=pytest.mark.xfail),
        pytest.param(operator.ne, marks=pytest.mark.xfail),
    ],
)
def test_df_sr_binop_col_order(op):
    colnames = [0, 1, 2]
    data = [[0, 2, 5], [3, None, 5], [6, 7, np.nan]]
    data = dict(zip(colnames, data, strict=True))

    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame.from_dict(data)

    gsr = cudf.Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"])
    psr = gsr.to_pandas()

    expect = op(pdf, psr).astype("float")
    out = op(gdf, gsr).astype("float")
    got = out[expect.columns]

    assert_eq(expect, got)


def test_different_shapes_and_columns(arithmetic_op):
    # Empty frame on the right side
    pd_frame = arithmetic_op(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({}))
    cd_frame = arithmetic_op(cudf.DataFrame({"x": [1, 2]}), cudf.DataFrame({}))
    assert_eq(cd_frame, pd_frame)

    # Empty frame on the left side
    pd_frame = pd.DataFrame({}) + pd.DataFrame({"x": [1, 2]})
    cd_frame = cudf.DataFrame({}) + cudf.DataFrame({"x": [1, 2]})
    assert_eq(cd_frame, pd_frame)

    # Note: the below rely on a discrepancy between cudf and pandas
    # While pandas inserts columns in alphabetical order, cudf inserts in the
    # order of whichever column comes first. So the following code will not
    # work if the names of columns are reversed i.e. ('y', 'x') != ('x', 'y')

    # More rows on the left side
    pd_frame = pd.DataFrame({"x": [1, 2, 3]}) + pd.DataFrame({"y": [1, 2]})
    cd_frame = cudf.DataFrame({"x": [1, 2, 3]}) + cudf.DataFrame({"y": [1, 2]})
    assert_eq(cd_frame, pd_frame)

    # More rows on the right side
    pd_frame = pd.DataFrame({"x": [1, 2]}) + pd.DataFrame({"y": [1, 2, 3]})
    cd_frame = cudf.DataFrame({"x": [1, 2]}) + cudf.DataFrame({"y": [1, 2, 3]})
    assert_eq(cd_frame, pd_frame)


def test_different_shapes_and_same_columns(arithmetic_op):
    pd_frame = arithmetic_op(
        pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [1, 2, 3]})
    )
    cd_frame = arithmetic_op(
        cudf.DataFrame({"x": [1, 2]}), cudf.DataFrame({"x": [1, 2, 3]})
    )
    # cast x as float64 so it matches pandas dtype
    cd_frame["x"] = cd_frame["x"].astype(np.float64)
    assert_eq(cd_frame, pd_frame)


def test_different_shapes_and_columns_with_unaligned_indices(arithmetic_op):
    # ``pow`` no longer overflows here: aligning the unaligned indices
    # introduces missing rows, which now promote the integer operands to
    # float64 (pandas semantics), so the pow is computed in float.

    # Test with a RangeIndex
    pdf1 = pd.DataFrame({"x": [4, 3, 2, 1], "y": [7, 3, 8, 6]})
    # Test with an Index
    pdf2 = pd.DataFrame(
        {"x": [1, 2, 3, 7], "y": [4, 5, 6, 7]}, index=[0, 1, 3, 4]
    )
    # Test with an Index in a different order
    pdf3 = pd.DataFrame(
        {"x": [4, 5, 6, 7], "y": [1, 2, 3, 7], "z": [0, 5, 3, 7]},
        index=[0, 3, 5, 3],
    )
    gdf1 = cudf.DataFrame(pdf1)
    gdf2 = cudf.DataFrame(pdf2)
    gdf3 = cudf.DataFrame(pdf3)

    pd_frame = arithmetic_op(arithmetic_op(pdf1, pdf2), pdf3)
    cd_frame = arithmetic_op(arithmetic_op(gdf1, gdf2), gdf3)
    # cast x and y as float64 so it matches pandas dtype
    cd_frame["x"] = cd_frame["x"].astype(np.float64)
    cd_frame["y"] = cd_frame["y"].astype(np.float64)

    # Sort both frames by index and then by all columns to ensure consistent ordering
    pd_sorted = pd_frame.sort_index().sort_values(list(pd_frame.columns))
    cd_sorted = cd_frame.sort_index().sort_values(list(cd_frame.columns))
    assert_eq(cd_sorted, pd_sorted)

    pdf1 = pd.DataFrame({"x": [1, 1]}, index=["a", "a"])
    pdf2 = pd.DataFrame({"x": [2]}, index=["a"])
    gdf1 = cudf.DataFrame(pdf1)
    gdf2 = cudf.DataFrame(pdf2)
    pd_frame = arithmetic_op(pdf1, pdf2)
    cd_frame = arithmetic_op(gdf1, gdf2)

    # Sort both frames consistently for comparison
    pd_sorted = pd_frame.sort_index().sort_values(list(pd_frame.columns))
    cd_sorted = cd_frame.sort_index().sort_values(list(cd_frame.columns))
    assert_eq(pd_sorted, cd_sorted)


@pytest.mark.parametrize(
    "pdf2",
    [
        pd.DataFrame({"a": [3, 2, 1]}, index=[3, 2, 1]),
        pd.DataFrame([3, 2]),
    ],
)
def test_df_different_index_shape(pdf2, comparison_op):
    df1 = cudf.DataFrame([1, 2, 3], index=[1, 2, 3])

    pdf1 = df1.to_pandas()
    df2 = cudf.DataFrame(pdf2)

    assert_exceptions_equal(
        lfunc=comparison_op,
        rfunc=comparison_op,
        lfunc_args_and_kwargs=([pdf1, pdf2],),
        rfunc_args_and_kwargs=([df1, df2],),
    )


@pytest.mark.parametrize("nulls", ["none", "some"])
@pytest.mark.parametrize("fill_value", [None, 27])
@pytest.mark.parametrize("other", ["df", "scalar"])
def test_operator_func_dataframe(
    arithmetic_op_method, nulls, fill_value, other
):
    num_rows = 100
    num_cols = 3

    def gen_df():
        rng = np.random.default_rng(seed=0)
        data = rng.random((num_rows, num_cols)) * 10000
        if nulls == "some":
            data.ravel()[
                rng.choice(
                    num_rows * num_cols,
                    size=int(num_rows * num_cols / 2),
                    replace=False,
                )
            ] = np.nan
        return pd.DataFrame(data, columns=["A", "B", "C"])

    pdf1 = gen_df()
    pdf2 = gen_df() if other == "df" else 59.0
    gdf1 = cudf.DataFrame(pdf1)
    gdf2 = cudf.DataFrame(pdf2) if other == "df" else 59.0

    got = getattr(gdf1, arithmetic_op_method)(gdf2, fill_value=fill_value)
    expect = getattr(pdf1, arithmetic_op_method)(pdf2, fill_value=fill_value)[
        list(got._data)
    ]

    assert_eq(expect, got)


@pytest.mark.parametrize("nulls", ["none", "some"])
@pytest.mark.parametrize("other", ["df", "scalar"])
def test_logical_operator_func_dataframe(comparison_op_method, nulls, other):
    num_rows = 100
    num_cols = 3

    def gen_df():
        rng = np.random.default_rng(seed=0)
        data = rng.random((num_rows, num_cols)) * 10000
        if nulls == "some":
            data.ravel()[
                rng.choice(
                    num_rows * num_cols,
                    size=int(num_rows * num_cols / 2),
                    replace=False,
                )
            ] = np.nan
        return pd.DataFrame(data, columns=["A", "B", "C"])

    pdf1 = gen_df()
    pdf2 = gen_df() if other == "df" else 59.0
    gdf1 = cudf.DataFrame(pdf1, nan_as_null=False)
    gdf2 = cudf.DataFrame(pdf2, nan_as_null=False) if other == "df" else 59.0

    got = getattr(gdf1, comparison_op_method)(gdf2)
    expect = getattr(pdf1, comparison_op_method)(pdf2)[list(got._data)]

    assert_eq(expect, got)


@pytest.mark.parametrize("data", [None, [-9, 7], [12, 18]])
@pytest.mark.parametrize("scalar", [1, 3, 12, np.nan])
def test_empty_column(binary_op, data, scalar):
    pdf = pd.DataFrame(columns=["a", "b"])
    if data is not None:
        pdf["a"] = data

    gdf = cudf.DataFrame(pdf)
    got = binary_op(gdf, scalar)
    expected = binary_op(pdf, scalar)

    if data is not None:
        expected = expected.fillna(None)
    # pandas can still hold an all NA columns with object dtype,
    # cuDF cannot, and uses StringDtype instead.
    assert_eq(expected, got, check_dtype=False)


@pytest.mark.parametrize(
    "df",
    [
        lambda: cudf.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 12, 13], [14, 15, 16, 17]]
        ),
        lambda: cudf.DataFrame(
            [
                [1.2, 2.3, 3.4, 4.5],
                [5.6, 6.7, 7.8, 8.9],
                [7.43, 4.2, 23.2, 23.2],
                [9.1, 2.4, 4.5, 65.34],
            ]
        ),
        lambda: cudf.Series([14, 15, 16, 17]),
        lambda: cudf.Series([14.15, 15.16, 16.17, 17.18]),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        lambda: cudf.DataFrame([[9, 10], [11, 12], [13, 14], [15, 16]]),
        lambda: cudf.DataFrame(
            [[9.4, 10.5], [11.6, 12.7], [13.8, 14.9], [15.1, 16.2]]
        ),
        lambda: cudf.Series([5, 6, 7, 8]),
        lambda: cudf.Series([5.6, 6.7, 7.8, 8.9]),
        lambda: np.array([5, 6, 7, 8]),
        # rapids-pre-commit-hooks: disable-next-line
        lambda: [25.5, 26.6, 27.7, 28.8],
    ],
)
def test_binops_dot(df, other):
    df = df()
    other = other()
    pdf = df.to_pandas()
    host_other = other.to_pandas() if hasattr(other, "to_pandas") else other

    expected = pdf @ host_other
    got = df @ other

    assert_eq(expected, got)


def test_binop_dot_preserve_index():
    ser = cudf.Series(range(2), index=["A", "B"])
    df = cudf.DataFrame(np.eye(2), columns=["A", "B"], index=["A", "B"])
    result = ser @ df
    expected = ser.to_pandas() @ df.to_pandas()
    assert_eq(result, expected)
