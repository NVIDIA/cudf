# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap

import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "pmi",
    [
        pd.MultiIndex.from_tuples(
            [(1, "red"), (1, "blue"), (2, "red"), (2, "blue")]
        ),
        pd.MultiIndex.from_tuples(
            [(1, "red"), (1, "blue"), (2, "red"), (2, "blue")] * 10
        ),
        pd.MultiIndex.from_tuples([(1, "red", 102, "sdf")]),
        pd.MultiIndex.from_tuples(
            [
                ("abc", 0.234, 1),
                ("a", -0.34, 0),
                ("ai", 111, 4385798),
                ("rapids", 0, 34534534),
            ],
            names=["alphabets", "floats", "ints"],
        ),
    ],
)
@pytest.mark.parametrize("max_seq_items", [None, 1, 2, 5, 10, 100])
def test_multiindex_repr(pmi, max_seq_items):
    with pd.option_context("display.max_seq_items", max_seq_items):
        gmi = cudf.from_pandas(pmi)

        assert repr(gmi) == repr(pmi)


@pytest.mark.parametrize(
    "gdi, expected_repr",
    [
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": [None, 1, 2, 3],
                        "b": ["abc", None, "xyz", None],
                        "c": [0.345, np.nan, 100, 10],
                    }
                )
                .set_index(["a", "b"])
                .index
            ),
            textwrap.dedent(
                """
                MultiIndex([(<NA>, 'abc'),
                            (   1,   nan),
                            (   2, 'xyz'),
                            (   3,   nan)],
                           names=['a', 'b'])
                """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": cudf.Series(
                            [None, np.nan, 2, 3], nan_as_null=False
                        ),
                        "b": ["abc", None, "xyz", None],
                        "c": [0.345, np.nan, 100, 10],
                    }
                )
                .set_index(["a", "b"])
                .index
            ),
            textwrap.dedent(
                """
            MultiIndex([(<NA>, 'abc'),
                        (<NA>,   nan),
                        ( 2.0, 'xyz'),
                        ( 3.0,   nan)],
                       names=['a', 'b'])
            """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": cudf.Series(
                            [None, 1, 2, 3], dtype="datetime64[ns]"
                        ),
                        "b": ["abc", None, "xyz", None],
                        "c": [0.345, np.nan, 100, 10],
                    }
                )
                .set_index(["a", "b"])
                .index
            ),
            textwrap.dedent(
                """
            MultiIndex([(                          'NaT', 'abc'),
                        ('1970-01-01 00:00:00.000000001',   nan),
                        ('1970-01-01 00:00:00.000000002', 'xyz'),
                        ('1970-01-01 00:00:00.000000003',   nan)],
                       names=['a', 'b'])
            """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": cudf.Series(
                            [None, 1, 2, 3], dtype="datetime64[ns]"
                        ),
                        "b": ["abc", None, "xyz", None],
                        "c": [0.345, np.nan, 100, 10],
                    }
                )
                .set_index(["a", "b", "c"])
                .index
            ),
            textwrap.dedent(
                """
                MultiIndex([(                          'NaT', 'abc', 0.345),
                            ('1970-01-01 00:00:00.000000001',   nan,  <NA>),
                            ('1970-01-01 00:00:00.000000002', 'xyz', 100.0),
                            ('1970-01-01 00:00:00.000000003',   nan,  10.0)],
                           names=['a', 'b', 'c'])
                """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": ["abc", None, "xyz", None],
                        "b": cudf.Series(
                            [None, 1, 2, 3], dtype="timedelta64[ns]"
                        ),
                        "c": [0.345, np.nan, 100, 10],
                    }
                )
                .set_index(["a", "b", "c"])
                .index
            ),
            textwrap.dedent(
                """
                MultiIndex([('abc',                         NaT, 0.345),
                            (  nan, '0 days 00:00:00.000000001',  <NA>),
                            ('xyz', '0 days 00:00:00.000000002', 100.0),
                            (  nan, '0 days 00:00:00.000000003',  10.0)],
                           names=['a', 'b', 'c'])
                """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": ["abc", None, "xyz", None],
                        "b": cudf.Series(
                            [None, 1, 2, 3], dtype="timedelta64[ns]"
                        ),
                        "c": [0.345, np.nan, 100, 10],
                    }
                )
                .set_index(["c", "a"])
                .index
            ),
            textwrap.dedent(
                """
                MultiIndex([(0.345, 'abc'),
                            ( <NA>,   nan),
                            (100.0, 'xyz'),
                            ( 10.0,   nan)],
                           names=['c', 'a'])
                """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": [None, None, None, None],
                        "b": cudf.Series(
                            [None, None, None, None], dtype="timedelta64[ns]"
                        ),
                        "c": [0.345, np.nan, 100, 10],
                    }
                )
                .set_index(["b", "a"])
                .index
            ),
            textwrap.dedent(
                """
            MultiIndex([(NaT, nan),
                        (NaT, nan),
                        (NaT, nan),
                        (NaT, nan)],
                       names=['b', 'a'])
            """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": [1, 2, None, 3, 5],
                        "b": [
                            "abc",
                            "def, hi, bye",
                            None,
                            ", one, two, three, four",
                            None,
                        ],
                        "c": cudf.Series(
                            [0.3232, np.nan, 1, None, -0.34534],
                            nan_as_null=False,
                        ),
                        "d": [None, 100, 2000324, None, None],
                    }
                )
                .set_index(["a", "b", "c", "d"])
                .index
            ),
            textwrap.dedent(
                """
    MultiIndex([(   1,                     'abc',   0.3232,    <NA>),
                (   2,            'def, hi, bye',     <NA>,     100),
                (<NA>,                       nan,      1.0, 2000324),
                (   3, ', one, two, three, four',     <NA>,    <NA>),
                (   5,                       nan, -0.34534,    <NA>)],
               names=['a', 'b', 'c', 'd'])
    """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": [1, 2, None, 3, 5],
                        "b": [
                            "abc",
                            "def, hi, bye",
                            None,
                            ", one, two, three, four",
                            None,
                        ],
                        "c": cudf.Series(
                            [0.3232, np.nan, 1, None, -0.34534],
                            nan_as_null=False,
                        ),
                        "d": [None, 100, 2000324, None, None],
                    }
                )
                .set_index(["b", "a", "c", "d"])
                .index
            ),
            textwrap.dedent(
                """
    MultiIndex([(                    'abc',    1,   0.3232,    <NA>),
                (           'def, hi, bye',    2,     <NA>,     100),
                (                      nan, <NA>,      1.0, 2000324),
                (', one, two, three, four',    3,     <NA>,    <NA>),
                (                      nan,    5, -0.34534,    <NA>)],
               names=['b', 'a', 'c', 'd'])
    """
            ),
        ),
        (
            lambda: (
                cudf.DataFrame(
                    {
                        "a": ["(abc", "2", None, "3", "5"],
                        "b": [
                            "abc",
                            "def, hi, bye",
                            None,
                            ", one, two, three, four",
                            None,
                        ],
                        "c": cudf.Series(
                            [0.3232, np.nan, 1, None, -0.34534],
                            nan_as_null=False,
                        ),
                        "d": [None, 100, 2000324, None, None],
                    }
                )
                .set_index(["a", "b", "c", "d"])
                .index
            ),
            textwrap.dedent(
                """
    MultiIndex([('(abc',                     'abc',   0.3232,    <NA>),
                (   '2',            'def, hi, bye',     <NA>,     100),
                (   nan,                       nan,      1.0, 2000324),
                (   '3', ', one, two, three, four',     <NA>,    <NA>),
                (   '5',                       nan, -0.34534,    <NA>)],
               names=['a', 'b', 'c', 'd'])
    """
            ),
        ),
    ],
)
def test_multiindex_null_repr(gdi, expected_repr):
    actual_repr = repr(gdi())

    assert actual_repr.split() == expected_repr.split()
