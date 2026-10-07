# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import cudf

_FULL_REPR_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "datetime64[ns]",
    "datetime64[us]",
    "datetime64[ms]",
    "datetime64[s]",
    "timedelta64[ns]",
    "timedelta64[us]",
    "timedelta64[ms]",
    "timedelta64[s]",
    "str",
    "bool",
    "category",
]


@pytest.fixture(
    scope="module",
    params=[(dtype, size) for dtype in _FULL_REPR_DTYPES for size in [20, 21]],
)
def full_dataframe(request):
    dtype, size = request.param
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {idx: rng.integers(0, 100, size) for idx in range(size)}
    ).astype(dtype)
    return pdf, cudf.from_pandas(pdf)


@pytest.mark.parametrize("ncols", [1, 2, 10])
def test_null_dataframe(ncols):
    dtype_categories = [
        "float32",
        "float64",
        "datetime64[ns]",
        "str",
        "category",
    ]
    rng = np.random.default_rng(seed=0)
    size = 20
    data = cudf.DataFrame()
    for dtype in dtype_categories:
        sr = cudf.Series(rng.integers(0, 128, size)).astype(dtype)
        sr[rng.choice([False, True], size=size)] = None
        data[dtype] = sr
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()
    with pd.option_context("display.max_columns", int(ncols)):
        pdf_repr = repr(pdf).replace("NaN", "<NA>").replace("None", "<NA>")
        gdf_repr = repr(gdf).replace("NaN", "<NA>")
        assert pdf_repr.split() == gdf_repr.split()


@pytest.mark.parametrize("nrows", [5, 10, 15])
@pytest.mark.parametrize("ncols", [5, 10, 15])
def test_full_dataframe_20(full_dataframe, nrows, ncols):
    pdf, gdf = full_dataframe

    with pd.option_context(
        "display.max_rows", int(nrows), "display.max_columns", int(ncols)
    ):
        assert repr(pdf) == repr(gdf)
        assert pdf._repr_html_() == gdf._repr_html_()
        assert pdf._repr_latex_() == gdf._repr_latex_()


@given(
    st.lists(
        st.integers(-9223372036854775808, 9223372036854775807),
        min_size=1,
        max_size=1000,
    )
)
@settings(deadline=None, max_examples=20)
def test_integer_dataframe(x):
    gdf = cudf.DataFrame({"x": x})
    pdf = gdf.to_pandas()
    with pd.option_context("display.max_columns", 1):
        assert repr(gdf) == repr(pdf)
        assert repr(gdf.T) == repr(pdf.T)


@given(st.lists(st.floats()))
@settings(deadline=None, max_examples=20)
def test_float_dataframe(x):
    gdf = cudf.DataFrame({"x": cudf.Series(x, dtype=float, nan_as_null=False)})
    pdf = gdf.to_pandas()
    assert repr(gdf) == repr(pdf)


def test_mixed_dataframe():
    data = {
        "Integer": np.array([2345, 11987, 9027, 9027]),
        "Date": np.array(
            ["18/04/1995", "14/07/1994", "07/06/2006", "16/09/2005"]
        ),
        "Float": np.array([9.001, 8.343, 6, 2.781]),
        "Integer2": np.array([2345, 106, 2088, 789277]),
        "Category": np.array(["M", "F", "F", "F"]),
        "String": np.array(["Alpha", "Beta", "Gamma", "Delta"]),
        "Boolean": np.array([True, False, True, False]),
    }
    mixed_gdf = cudf.DataFrame(data)
    mixed_pdf = pd.DataFrame(data)
    assert repr(mixed_gdf) == repr(mixed_pdf)


def test_MI():
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "a": rng.integers(0, 4, 10),
            "b": rng.integers(0, 4, 10),
            "c": rng.integers(0, 4, 10),
        }
    )
    levels = [["a", "b", "c", "d"], ["w", "x", "y", "z"], ["m", "n"]]
    codes = [
        [0, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]
    with pd.option_context("display.max_rows", 999, "display.max_columns", 0):
        gdf = gdf.set_index(cudf.MultiIndex(levels=levels, codes=codes))
        pdf = gdf.to_pandas()
        assert repr(gdf) == repr(pdf)
        assert repr(gdf.index) == repr(pdf.index)
        assert repr(gdf.T) == repr(pdf.T)


@pytest.mark.parametrize("nrows", [0, 1, 3, 5, 10])
@pytest.mark.parametrize("ncols", [0, 1, 2, 3])
def test_groupby_MI(nrows, ncols):
    gdf = cudf.DataFrame(
        {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["a", "b"], sort=True).count()
    pdg = pdf.groupby(["a", "b"], sort=True).count()
    with pd.option_context(
        "display.max_rows", nrows, "display.max_columns", ncols
    ):
        assert repr(gdg) == repr(pdg)
        assert repr(gdg.index) == repr(pdg.index)
        assert repr(gdg.T) == repr(pdg.T)


@pytest.fixture(
    scope="module",
    params=[
        lambda: cudf.DataFrame({"a": range(10000)}),
        lambda: cudf.DataFrame({"a": range(10000), "b": range(10000)}),
        lambda: cudf.DataFrame({"a": range(20), "b": range(20)}),
        lambda: cudf.DataFrame(
            {
                "a": range(20),
                "b": range(20),
                "c": ["abc", "def", "xyz", "def", "pqr"] * 4,
            }
        ),
        lambda: cudf.DataFrame(index=[1, 2, 3]),
        lambda: cudf.DataFrame(index=range(10000)),
        lambda: cudf.DataFrame(columns=["a", "b", "c", "d"]),
        lambda: cudf.DataFrame(columns=["a"], index=range(10000)),
        lambda: cudf.DataFrame(
            columns=["a", "col2", "...col n"], index=range(10000)
        ),
        lambda: cudf.DataFrame(index=cudf.Series(range(10000)).astype("str")),
        lambda: cudf.DataFrame(
            columns=["a", "b", "c", "d"],
            index=cudf.Series(range(10000)).astype("str"),
        ),
    ],
)
def sliced_dataframe(request):
    return request.param()


@pytest.mark.parametrize(
    "slc",
    [
        slice(2500, 5000),
        slice(2500, 2501),
        slice(5000),
        slice(1, 10),
        slice(10, 20),
        slice(15, 2400),
    ],
)
@pytest.mark.parametrize("max_seq_items", [1, 10, 60, 10000, None])
@pytest.mark.parametrize("max_rows", [1, 10, 60, 10000, None])
def test_dataframe_sliced(sliced_dataframe, slc, max_seq_items, max_rows):
    with pd.option_context(
        "display.max_seq_items", max_seq_items, "display.max_rows", max_rows
    ):
        pdf = sliced_dataframe.to_pandas()

        sliced_gdf = sliced_dataframe[slc]
        sliced_pdf = pdf[slc]

        expected_repr = repr(sliced_pdf)
        actual_repr = repr(sliced_gdf)

        assert expected_repr == actual_repr


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, None]),
        pd.DataFrame(
            {
                "a": [1, None, 3],
                "string_col": ["hello", "world", "rapids"],
            },
            index=[None, "a", "b"],
        ),
        pd.DataFrame([], index=[None, "a", "b"]),
        pd.DataFrame({"aa": [None, None]}, index=[None, None]),
        pd.DataFrame({"aa": [1, 2, 3]}, index=[None, None, None]),
        pd.DataFrame(
            {"aa": [None, 2, 3]},
            index=np.array([1, None, None], dtype="datetime64[ns]"),
        ),
        pd.DataFrame(
            {"aa": [None, 2, 3]},
            index=np.array([100, None, None], dtype="datetime64[ns]"),
        ),
        pd.DataFrame(
            {"aa": [None, None, None]},
            index=np.array([None, None, None], dtype="datetime64[ns]"),
        ),
        pd.DataFrame(
            {"aa": [1, None, 3]},
            index=np.array([10, 15, None], dtype="datetime64[ns]"),
        ),
        pd.DataFrame(
            {"a": [1, 2, None], "v": [10, None, 22], "p": [100, 200, 300]}
        ).set_index(["a", "v"]),
        pd.DataFrame(
            {
                "a": [1, 2, None],
                "v": ["n", "c", "a"],
                "p": [None, None, None],
            }
        ).set_index(["a", "v"]),
        pd.DataFrame(
            {
                "a": np.array([1, None, None], dtype="datetime64[ns]"),
                "v": ["n", "c", "a"],
                "p": [None, None, None],
            }
        ).set_index(["a", "v"]),
    ],
)
def test_dataframe_null_index_repr(df):
    with cudf.option_context("mode.pandas_compatible", True):
        pdf = df
        gdf = cudf.from_pandas(pdf)

        expected_repr = repr(pdf)
        actual_repr = repr(gdf)

    assert expected_repr.split() == actual_repr.split()


@pytest.mark.parametrize(
    "df",
    [
        lambda: cudf.DataFrame(
            {
                "a": cudf.Series(
                    [1000000, 200000, 3000000], dtype="timedelta64[s]"
                )
            }
        ),
        lambda: cudf.DataFrame(
            {
                "a": cudf.Series(
                    [
                        136457654,
                        None,
                        245345345,
                        223432411,
                        None,
                        3634548734,
                        23234,
                    ],
                    dtype="timedelta64[s]",
                ),
                "b": [10, 11, 22, 33, 44, 55, 66],
            }
        ),
        lambda: cudf.DataFrame(
            {
                "a": cudf.Series(
                    [
                        136457654,
                        None,
                        245345345,
                        223432411,
                        None,
                        3634548734,
                        23234,
                    ],
                    dtype="timedelta64[s]",
                    index=["a", "b", "c", "d", "e", "f", "g"],
                )
            }
        ),
        lambda: cudf.DataFrame(
            {
                "a": cudf.Series(
                    [1, 2, 3, 4, 5, 6, 7],
                    index=cudf.Index(
                        [
                            136457654,
                            None,
                            245345345,
                            223432411,
                            None,
                            3634548734,
                            23234,
                        ],
                        dtype="timedelta64[ms]",
                    ),
                )
            }
        ),
        lambda: cudf.DataFrame(
            {
                "a": cudf.Series(
                    ["a", "f", "q", "e", "w", "e", "t"],
                    index=cudf.Index(
                        [
                            136457654,
                            None,
                            245345345,
                            223432411,
                            None,
                            3634548734,
                            23234,
                        ],
                        dtype="timedelta64[ns]",
                    ),
                )
            }
        ),
    ],
)
def test_timedelta_dataframe_repr(df):
    gdf = df()
    pdf = gdf.to_pandas()
    assert repr(gdf).split() == repr(pdf).split()


def test_categorical_dataframe_with_nan_repr():
    series = cudf.Series(
        [1, 2, np.nan, 10, np.nan, None], nan_as_null=False
    ).astype("category")
    df = cudf.DataFrame({"a": series})
    expected_repr = textwrap.dedent(
        """
          a
    0   1.0
    1   2.0
    2  <NA>
    3  10.0
    4  <NA>
    5  <NA>
    """
    )

    assert repr(df).split() == expected_repr.split()


def test_repr_struct_after_concat():
    df = cudf.DataFrame(
        {
            "a": cudf.Series(
                [
                    {"sa": 2056831253},
                    {"sa": -1463792165},
                    {"sa": 1735783038},
                    {"sa": 103774433},
                    {"sa": -1413247520},
                ]
                * 13
            ),
            "b": cudf.Series(
                [
                    {"sa": {"ssa": 1140062029}},
                    None,
                    {"sa": {"ssa": 1998862860}},
                    {"sa": None},
                    {"sa": {"ssa": -395088502}},
                ]
                * 13
            ),
        }
    )
    pdf = df.to_pandas()

    assert repr(df) == repr(pdf)
