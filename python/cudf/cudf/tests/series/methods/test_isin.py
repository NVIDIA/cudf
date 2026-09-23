# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [],
        [0, 12, 14],
        [0, 14, 12, 12, 3, 10, 12, 14],
        np.random.default_rng(seed=0).integers(-100, 100, 200),
        pd.Series([0.0, 1.0, None, 10.0]),
        [None, None, None, None],
        [np.nan, None, -1, 2, 3],
        [1, 2],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.random.default_rng(seed=0).integers(-100, 100, 10),
        [],
        [np.nan, None, -1, 2, 3],
        [1.0, 12.0, None, None, 120],
        [0.1, 12.1, 14.1],
        [0, 14, 12, 12, 3, 10, 12, 14, None],
        [None, None, None],
        ["0", "12", "14"],
        ["0", "12", "14", "a"],
        [1.0, 2.5],
    ],
)
def test_isin_numeric(data, values):
    rng = np.random.default_rng(seed=0)
    index = rng.integers(0, 100, len(data))
    psr = pd.Series(data, index=index)
    gsr = cudf.Series(psr, nan_as_null=False)

    expected = psr.isin(values)
    got = gsr.isin(values)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(
            ["2018-01-01", "2019-04-03", None, "2019-12-30"],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                "2018-01-01",
                "2019-04-03",
                None,
                "2019-12-30",
                "2018-01-01",
                "2018-01-01",
            ],
            dtype="datetime64[ns]",
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        [1514764800000000000, 1577664000000000000],
        [
            1514764800000000000,
            1577664000000000000,
            1577664000000000000,
            1577664000000000000,
            1514764800000000000,
        ],
        ["2019-04-03", "2019-12-30", "2012-01-01"],
        [
            "2012-01-01",
            "2012-01-01",
            "2012-01-01",
            "2019-04-03",
            "2019-12-30",
            "2012-01-01",
        ],
    ],
)
def test_isin_datetime(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        ["this", "is", None, "a", "test"],
        ["test", "this", "test", "is", None, "test", "a", "test"],
        ["0", "12", "14"],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [None, None, None],
        ["12", "14", "19"],
        [12, 14, 19],
        ["is", "this", "is", "this", "is"],
    ],
)
def test_isin_string(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(["a", "b", "c", "c", "c", "d", "e"], dtype="category"),
        pd.Series(["a", "b", None, "c", "d", "e"], dtype="category"),
        pd.Series([0, 3, 10, 12], dtype="category"),
        pd.Series([0, 3, 10, 12, 0, 10, 3, 0, 0, 3, 3], dtype="category"),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a", "b", None, "f", "words"],
        ["0", "12", None, "14"],
        [0, 10, 12, None, 39, 40, 1000],
        [0, 0, 0, 0, 3, 3, 3, None, 1, 2, 3],
    ],
)
def test_isin_categorical(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
@pytest.mark.parametrize(
    "data,values",
    [
        ([0, 1, 0], [1]),
        ([0, 1, 0], [1, pd.NA]),
        ([0, pd.NA, 0], [1, 0]),
        ([0, 1, pd.NA], [1, pd.NA]),
        ([0, 1, pd.NA], [1, np.nan]),
        ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None]),
    ],
)
def test_isin_masked_types(dtype, data, values):
    # Series.isin on a pandas masked (nullable integer/float/boolean) dtype
    # returns a nullable BooleanDtype result and matches pandas' NA semantics:
    #  * comparison is done on the underlying values (a boolean element equals
    #    the integer 1), and
    #  * an NA element is considered present only when pd.NA itself is one of
    #    ``values`` (a plain NaN/None/NaT does not match).
    psr = pd.Series(data, dtype=dtype)
    gsr = cudf.Series(data, dtype=dtype)

    got = gsr.isin(values)
    expected = psr.isin(values)

    assert got.dtype == pd.BooleanDtype()
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "values",
    [[1], [0], [1, 0], [2], [1.0], [1.5], [True], [False]],
)
def test_isin_bool_against_numeric(values):
    # A boolean Series compares equal to the integers/floats 0 and 1, matching
    # numpy/pandas value semantics (previously cudf returned all-False for a
    # numeric ``values`` argument).
    psr = pd.Series([True, False, False, True])
    gsr = cudf.Series([True, False, False, True])

    got = gsr.isin(values)
    assert got.dtype == np.dtype("bool")
    assert_eq(got, psr.isin(values))


@pytest.mark.parametrize(
    "values",
    [
        lambda: (value for value in [1, 2]),
        lambda: cudf.Series([1, 2]),
        lambda: cudf.Index([1, 2]),
        lambda: cp.array([1, 2]),
        lambda: pa.chunked_array([[1, 2]]),
    ],
    ids=["generator", "cudf-series", "cudf-index", "cupy", "pyarrow"],
)
def test_isin_masked_values_containers(values):
    # ``values`` is materialized on host once for masked dtypes:
    # device-backed inputs are moved in a single transfer instead of
    # element-wise reads (cudf objects are not Python-iterable at all).
    psr = pd.Series([0, 1, pd.NA], dtype="Int64")
    gsr = cudf.Series([0, 1, pd.NA], dtype="Int64")

    assert_eq(gsr.isin(values()), psr.isin([1, 2]))


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1, pd.NA], dtype="Int64"),
        pd.Index([1, pd.NA], dtype="Int64"),
        pd.array([1, pd.NA], dtype="Int64"),
    ],
    ids=["pd-series", "pd-index", "pd-array"],
)
def test_isin_masked_values_container_NA_does_not_match(values):
    # pandas materializes pandas-container ``values`` with np.asarray, where
    # pd.NA decays to NaN: an NA needle inside a masked container does NOT
    # match NA rows, unlike a literal pd.NA in a list.
    psr = pd.Series([0, 1, pd.NA], dtype="Int64")
    gsr = cudf.Series([0, 1, pd.NA], dtype="Int64")

    assert_eq(gsr.isin(values), psr.isin(values))


def test_isin_masked_one_shot_iterator_with_NA():
    # A one-shot iterator must not be exhausted before the pd.NA
    # inspection: the NA row still matches when pd.NA is one of ``values``.
    psr = pd.Series([0, 1, pd.NA], dtype="Int64")
    gsr = cudf.Series([0, 1, pd.NA], dtype="Int64")

    got = gsr.isin(value for value in [1, pd.NA])
    expected = psr.isin([1, pd.NA])
    assert_eq(got, expected)


def test_isin_masked_nan_is_value_not_na():
    # A genuine NaN value in a masked float column is data, not NA: it
    # matches a NaN needle and does not match pd.NA (mirrors pandas, where
    # only the mask is NA).
    psr = pd.Series(
        pd.arrays.FloatingArray(
            np.array([np.nan, 1.0, 0.0]),
            np.array([False, False, True]),
        )
    )
    gsr = cudf.Series([np.nan, 1.0, None], dtype="Float64", nan_as_null=False)

    assert_eq(gsr.isin([np.nan]), psr.isin([np.nan]))
    assert_eq(gsr.isin([pd.NA]), psr.isin([pd.NA]))


def test_isin_masked_pandas_compatible_mode():
    # The masked path must work in pandas-compatible mode (the
    # masked-with-nulls to numpy astype guard must not trigger) and must
    # not mutate the input Series' dtype via the in-place astype
    # short-circuit.
    with cudf.option_context("mode.pandas_compatible", True):
        gsr = cudf.Series([1, pd.NA], dtype="Int64")
        got = gsr.isin([1])
        assert gsr.dtype == pd.Int64Dtype()
        assert got.dtype == pd.BooleanDtype()
        assert got.to_pandas().tolist() == [True, False]

        gsr = cudf.Series([True, pd.NA], dtype="boolean")
        got = gsr.isin([True])
        assert gsr.dtype == pd.BooleanDtype()
        assert got.to_pandas().tolist() == [True, False]

        gsr = cudf.Series([1.5, 2.5], dtype="Float64")
        got = gsr.isin([1.5])
        assert gsr.dtype == pd.Float64Dtype()
        assert got.to_pandas().tolist() == [True, False]


@pytest.mark.parametrize("values", [[], [pd.NA], [None], [1]])
def test_isin_masked_all_na(values):
    # An all-NA masked Series with needles that clean to empty used to
    # raise in pandas-compatible mode (an empty needle list produces an
    # object-dtype column that an all-null column cannot be cast to there).
    psr = pd.Series([pd.NA, pd.NA], dtype="Int64")
    expected = psr.isin(values)

    gsr = cudf.Series([pd.NA, pd.NA], dtype="Int64")
    assert_eq(gsr.isin(values), expected)

    with cudf.option_context("mode.pandas_compatible", True):
        gsr = cudf.Series([pd.NA, pd.NA], dtype="Int64")
        got = gsr.isin(values)
    assert_eq(got, expected)


def test_isin_masked_does_not_mutate_dtype():
    # Regression test: the astype in the masked path used to flip the
    # input's dtype from Int64 to int64 in pandas-compatible mode.
    gsr = cudf.Series([1, 2], dtype="Int64")
    gsr.isin([1])
    assert gsr.dtype == pd.Int64Dtype()

    with cudf.option_context("mode.pandas_compatible", True):
        gsr = cudf.Series([1, 2], dtype="Int64")
        gsr.isin([1])
        assert gsr.dtype == pd.Int64Dtype()


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([0, 1, 0], dtype=pd.ArrowDtype(pa.int64())),
        pd.Series([1.0, 2.0, None], dtype=pd.ArrowDtype(pa.float64())),
        pd.Series([True, False, True], dtype=pd.ArrowDtype(pa.bool_())),
        pd.Series(["a", "b", "a"], dtype="category"),
    ],
)
def test_isin_non_masked_extension_returns_numpy_bool(psr):
    # Arrow and categorical inputs yield a numpy bool result (only masked
    # numeric/boolean dtypes upgrade to nullable boolean).
    gsr = cudf.from_pandas(psr)

    got = gsr.isin([psr.iloc[0]])
    assert got.dtype == np.dtype("bool")
    assert_eq(got, psr.isin([psr.iloc[0]]))


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("object"),
        pd.StringDtype(storage="python", na_value=pd.NA),
        pd.StringDtype(storage="python", na_value=np.nan),
        pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        pd.StringDtype(storage="pyarrow", na_value=np.nan),
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        ["b", "x"],
        np.array(["b", "x"], dtype=object),
        pd.array(
            ["b", "x"], dtype=pd.StringDtype(storage="python", na_value=pd.NA)
        ),
        pd.array(
            ["b", "x"],
            dtype=pd.StringDtype(storage="python", na_value=np.nan),
        ),
        pd.array(
            ["b", "x"],
            dtype=pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        ),
        pd.array(
            ["b", "x"],
            dtype=pd.StringDtype(storage="pyarrow", na_value=np.nan),
        ),
        pd.array(["b", "x"], dtype=pd.ArrowDtype(pa.string())),
        pa.array(["b", "x"]),
    ],
)
def test_isin_string_dtype_flavors(dtype, values):
    # isin must match on element values regardless of which string dtype
    # flavor the series and the values use (object, pd.StringDtype with
    # python/pyarrow storage and NaN/NA na_value, pd.ArrowDtype string).
    data = ["a", "b", "c"]
    psr = pd.Series(data, dtype=dtype)
    gsr = cudf.Series(data, dtype=dtype)

    got = gsr.isin(values)
    if isinstance(values, pa.Array) and (
        dtype == np.dtype("object")
        or (isinstance(dtype, pd.StringDtype) and dtype.storage == "python")
    ):
        # pandas does not match pyarrow.Array values against
        # object/python-storage series (returns all-False); cudf matches
        # on element values.
        assert got.to_pandas().tolist() == [False, True, False]
    else:
        expected = psr.isin(values)
        assert_eq(got, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("object"),
        pd.StringDtype(storage="python", na_value=pd.NA),
        pd.StringDtype(storage="python", na_value=np.nan),
        pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        pd.StringDtype(storage="pyarrow", na_value=np.nan),
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
    ],
)
def test_isin_string_null_values(dtype):
    # cudf treats every NA-like value in ``values`` (None, pd.NA) as a
    # match for nulls in the series. Results are compared with pandas
    # except where pandas diverges: object dtype distinguishes the
    # stored None from pd.NA, and ArrowDtype raises ArrowTypeError when
    # the value set contains nulls.
    data = ["a", "b", None]
    psr = pd.Series(data, dtype=dtype)
    gsr = cudf.Series(data, dtype=dtype)

    got = gsr.isin([None])
    if isinstance(dtype, pd.ArrowDtype):
        assert got.to_pandas().tolist() == [False, False, True]
    else:
        assert_eq(got, psr.isin([None]))

    got = gsr.isin(["a", pd.NA])
    if isinstance(dtype, pd.ArrowDtype) or dtype == np.dtype("object"):
        assert got.to_pandas().tolist() == [True, False, True]
    else:
        assert_eq(got, psr.isin(["a", pd.NA]))

    assert_eq(gsr.isin([]), psr.isin([]))
