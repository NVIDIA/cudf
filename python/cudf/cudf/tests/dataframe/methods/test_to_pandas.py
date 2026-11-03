# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import datetime
import decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


def test_to_pandas():
    df = cudf.DataFrame(
        {
            "a": np.arange(5, dtype=np.int32),
            "b": np.arange(10, 15, dtype=np.float64),
            "c": np.array([True, False, None, True, True]),
        }
    )

    pdf = df.to_pandas()

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df["a"].dtype == pdf["a"].dtype
    assert df["b"].dtype == pdf["b"].dtype

    # Notice, the dtype differ when Pandas and cudf boolean series
    # contains None/NaN
    assert df["c"].dtype == np.bool_
    assert pdf["c"].dtype == np.object_

    assert len(df["a"]) == len(pdf["a"])
    assert len(df["b"]) == len(pdf["b"])
    assert len(df["c"]) == len(pdf["c"])


def test_list_to_pandas_nullable_true():
    df = cudf.DataFrame({"a": cudf.Series([[1, 2, 3]])})
    with pytest.raises(NotImplementedError):
        df.to_pandas(nullable=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_dataframe_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    df = cudf.DataFrame({"a": pa_array})
    result = df.to_pandas(arrow_type=True)
    expected = pd.DataFrame({"a": pd.arrays.ArrowExtensionArray(pa_array)})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_dataframe_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = pa.array([scalar, None])
    df = cudf.DataFrame({"a": pa_array})
    with pytest.raises(ValueError):
        df.to_pandas(nullable=True, arrow_type=True)


@pytest.mark.parametrize(
    "df,expected_pdf",
    [
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series([1, 2, None, 3], dtype="uint8"),
                    "b": cudf.Series([23, None, None, 32], dtype="uint16"),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series([1, 2, None, 3], dtype=pd.UInt8Dtype()),
                    "b": pd.Series(
                        [23, None, None, 32], dtype=pd.UInt16Dtype()
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series([None, 123, None, 1], dtype="uint32"),
                    "b": cudf.Series(
                        [234, 2323, 23432, None, None, 224], dtype="uint64"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [None, 123, None, 1], dtype=pd.UInt32Dtype()
                    ),
                    "b": pd.Series(
                        [234, 2323, 23432, None, None, 224],
                        dtype=pd.UInt64Dtype(),
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [-10, 1, None, -1, None, 3], dtype="int8"
                    ),
                    "b": cudf.Series(
                        [111, None, 222, None, 13], dtype="int16"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [-10, 1, None, -1, None, 3], dtype=pd.Int8Dtype()
                    ),
                    "b": pd.Series(
                        [111, None, 222, None, 13], dtype=pd.Int16Dtype()
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [11, None, 22, 33, None, 2, None, 3], dtype="int32"
                    ),
                    "b": cudf.Series(
                        [32431, None, None, 32322, 0, 10, -32324, None],
                        dtype="int64",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [11, None, 22, 33, None, 2, None, 3],
                        dtype=pd.Int32Dtype(),
                    ),
                    "b": pd.Series(
                        [32431, None, None, 32322, 0, 10, -32324, None],
                        dtype=pd.Int64Dtype(),
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [True, None, False, None, False, True, True, False],
                        dtype="bool_",
                    ),
                    "b": cudf.Series(
                        [
                            "abc",
                            "a",
                            None,
                            "hello world",
                            "foo buzz",
                            "",
                            None,
                            "rapids ai",
                        ],
                    ),
                    "c": cudf.Series(
                        [0.1, None, 0.2, None, 3, 4, 1000, None],
                        dtype="float64",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [True, None, False, None, False, True, True, False],
                        dtype=pd.BooleanDtype(),
                    ),
                    "b": pd.Series(
                        [
                            "abc",
                            "a",
                            None,
                            "hello world",
                            "foo buzz",
                            "",
                            None,
                            "rapids ai",
                        ],
                        dtype=pd.StringDtype(na_value=np.nan),
                    ),
                    "c": pd.Series(
                        [0.1, None, 0.2, None, 3, 4, 1000, None],
                        dtype=pd.Float64Dtype(),
                    ),
                }
            ),
        ),
    ],
)
def test_dataframe_to_pandas_nullable_dtypes(df, expected_pdf):
    actual_pdf = df().to_pandas(nullable=True)

    assert_eq(actual_pdf, expected_pdf)
