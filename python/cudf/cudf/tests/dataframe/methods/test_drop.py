# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "c": range(1, 11)},
            index=pd.Index(
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                name="custom_name",
            ),
        ),
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5}
        ),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [["a"], ["b"], "a", "b", ["a", "b"]],
)
def test_dataframe_drop_columns(pdf, columns, inplace):
    if inplace:
        pdf = pdf.copy()
    gdf = cudf.from_pandas(pdf)

    expected = pdf.drop(columns=columns)
    actual = gdf.drop(columns=columns, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.parametrize("obj", ["Index", "Series"])
def test_drop_cudf_obj_columns(obj):
    pdf = pd.DataFrame({"A": [1], "B": [1]})
    gdf = cudf.from_pandas(pdf)

    columns = ["B"]
    expected = pdf.drop(labels=getattr(pd, obj)(columns), axis=1)
    actual = gdf.drop(columns=getattr(cudf, obj)(columns), axis=1)
    assert_eq(expected, actual)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "c": range(1, 11)},
            index=pd.Index(list(range(10)), name="custom_name"),
        ),
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5}
        ),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [
        [1],
        [0],
        1,
        5,
        [5, 9],
        pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        pd.Index([0, 1, 8, 9], name="new name"),
    ],
)
def test_dataframe_drop_labels_axis_0(pdf, labels, inplace):
    pdf = pdf.copy()
    gdf = cudf.from_pandas(pdf)

    expected = pdf.drop(labels=labels, axis=0)
    actual = gdf.drop(labels=labels, axis=0, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame({"a": range(10), "b": range(10, 20), "c": range(1, 11)}),
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5}
        ),
        pd.DataFrame(
            {
                "a": range(10),
                "b": range(10, 20),
            },
            index=pd.Index(list(range(10)), dtype="uint64"),
        ),
    ],
)
@pytest.mark.parametrize(
    "index",
    [[1], [0], 1, 5, [5, 9], pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_index(pdf, index, inplace):
    pdf = pdf.copy()
    gdf = cudf.from_pandas(pdf)

    expected = pdf.drop(index=index)
    actual = gdf.drop(index=index, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize(
    "index,level",
    [
        ("cow", 0),
        ("lama", 0),
        ("falcon", 0),
        ("speed", 1),
        ("weight", 1),
        ("length", 1),
        ("cow", None),
        (
            "lama",
            None,
        ),
        (
            "falcon",
            None,
        ),
    ],
)
def test_dataframe_drop_multiindex(index, level, inplace):
    pdf = pd.DataFrame(
        {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5},
        index=pd.MultiIndex(
            levels=[
                ["lama", "cow", "falcon"],
                ["speed", "weight", "length"],
            ],
            codes=[
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
            ],
        ),
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.drop(index=index, level=level)
    actual = gdf.drop(index=index, inplace=inplace, level=level)

    if inplace:
        assert actual is None
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize("data", [{"c": range(1, 11)}, {"d": ["a", "v"] * 5}])
@pytest.mark.parametrize("labels", [["a"], ["b"], "a", "b", ["a", "b"]])
def test_dataframe_drop_labels_axis_1(data, labels, inplace):
    pdf = pd.DataFrame({"a": range(10), "b": range(10, 20), **data})
    gdf = cudf.from_pandas(pdf)

    expected = pdf.drop(labels=labels, axis=1)
    actual = gdf.drop(labels=labels, axis=1, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gdf

    assert_eq(expected, actual)


def test_dataframe_drop_error():
    df = cudf.DataFrame({"a": [1], "b": [2], "c": [3]})
    pdf = df.to_pandas()

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"columns": "d"}),
        rfunc_args_and_kwargs=([], {"columns": "d"}),
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"columns": ["a", "d", "b"]}),
        rfunc_args_and_kwargs=([], {"columns": ["a", "d", "b"]}),
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
        rfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"axis": 1}),
        rfunc_args_and_kwargs=([], {"axis": 1}),
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([[2, 0]],),
        rfunc_args_and_kwargs=([[2, 0]],),
    )


def test_dataframe_drop_raises():
    df = cudf.DataFrame(
        {"a": [1, 2, 3], "c": [10, 20, 30]}, index=["x", "y", "z"]
    )
    pdf = df.to_pandas()
    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=(["p"],),
        rfunc_args_and_kwargs=(["p"],),
    )

    # label dtype mismatch
    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([3],),
        rfunc_args_and_kwargs=([3],),
    )

    expect = pdf.drop("p", errors="ignore")
    actual = df.drop("p", errors="ignore")

    assert_eq(actual, expect)

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"columns": "p"}),
        rfunc_args_and_kwargs=([], {"columns": "p"}),
    )

    expect = pdf.drop(columns="p", errors="ignore")
    actual = df.drop(columns="p", errors="ignore")

    assert_eq(actual, expect)

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"labels": "p", "axis": 1}),
        rfunc_args_and_kwargs=([], {"labels": "p", "axis": 1}),
    )

    expect = pdf.drop(labels="p", axis=1, errors="ignore")
    actual = df.drop(labels="p", axis=1, errors="ignore")

    assert_eq(actual, expect)
