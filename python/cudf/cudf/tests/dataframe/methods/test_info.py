# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import io
import textwrap

import numpy as np
import pandas as pd

import cudf


def test_dataframe_info_basic():
    buffer = io.StringIO()
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    Index: 10 entries, a to 1111
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   0       10 non-null     float64
     1   1       10 non-null     float64
     2   2       10 non-null     float64
     3   3       10 non-null     float64
     4   4       10 non-null     float64
     5   5       10 non-null     float64
     6   6       10 non-null     float64
     7   7       10 non-null     float64
     8   8       10 non-null     float64
     9   9       10 non-null     float64
    dtypes: float64(10)
    memory usage: 903.0 bytes
    """
    )
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.standard_normal(size=(10, 10)),
        index=["a", "2", "3", "4", "5", "6", "7", "8", "100", "1111"],
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    assert str_cmp == s


def test_dataframe_info_verbose_mem_usage():
    buffer = io.StringIO()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["safdas", "assa", "asdasd"]})
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       3 non-null      int64
     1   b       3 non-null      str
    dtypes: int64(1), str(1)
    memory usage: 72.0 bytes
    """
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Columns: 2 entries, a to b
    dtypes: int64(1), str(1)
    memory usage: 72.0 bytes
    """
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=False)
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["safdas", "assa", "asdasd"]},
        index=["sdfdsf", "sdfsdfds", "dsfdf"],
    )
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    Index: 3 entries, sdfdsf to dsfdf
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       3 non-null      int64
     1   b       3 non-null      str
    dtypes: int64(1), str(1)
    memory usage: 123.0 bytes
    """
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=True, memory_usage="deep")
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    int_values = [1, 2, 3, 4, 5]
    text_values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    float_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    df = cudf.DataFrame(
        {
            "int_col": int_values,
            "text_col": text_values,
            "float_col": float_values,
        }
    )
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   int_col    5 non-null      int64
     1   text_col   5 non-null      str
     2   float_col  5 non-null      float64
    dtypes: float64(1), int64(1), str(1)
    memory usage: 130.0 bytes
    """
    )
    df.info(buf=buffer, verbose=True, memory_usage="deep")
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)


def test_dataframe_info_null_counts():
    int_values = [1, 2, 3, 4, 5]
    text_values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    float_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    df = cudf.DataFrame(
        {
            "int_col": int_values,
            "text_col": text_values,
            "float_col": float_values,
        }
    )
    buffer = io.StringIO()
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Dtype
    ---  ------     -----
     0   int_col    int64
     1   text_col   str
     2   float_col  float64
    dtypes: float64(1), int64(1), str(1)
    memory usage: 130.0 bytes
    """
    )
    df.info(buf=buffer, verbose=True, null_counts=False)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df.info(buf=buffer, verbose=True, max_cols=0)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df = cudf.DataFrame()

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 0 entries
    Empty DataFrame"""
    )
    df.info(buf=buffer, verbose=True)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, None, 10, 11, 12, None],
            "b": ["a", "b", "c", "sd", "sdf", "sd", None, None],
        }
    )

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 2 columns):
     #   Column  Dtype
    ---  ------  -----
     0   a       int64
     1   b       str
    dtypes: int64(1), str(1)
    memory usage: 238.0 bytes
    """
    )
    with pd.option_context("display.max_info_rows", 2):
        df.info(buf=buffer, max_cols=2, null_counts=None)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       6 non-null      int64
     1   b       6 non-null      str
    dtypes: int64(1), str(1)
    memory usage: 238.0 bytes
    """
    )

    df.info(buf=buffer, max_cols=2, null_counts=None)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df.info(buf=buffer, null_counts=True)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string
