# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.fixture(params=["outer", "inner"])
def join(request):
    return request.param


def make_frames(index=None, nulls="none"):
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": list("abcde") * 2,
        }
    )
    df.z = df.z.astype("category")
    df2 = pd.DataFrame(
        {
            "x": range(10, 20),
            "y": list(map(float, range(10, 20))),
            "z": list("edcba") * 2,
        }
    )
    df2.z = df2.z.astype("category")
    if nulls == "all":
        df.y = np.full_like(df.y, np.nan)
        df2.y = np.full_like(df2.y, np.nan)
    if nulls == "some":
        mask = np.arange(10)
        rng.shuffle(mask)
        mask = mask[:5]
        df.loc[mask, "y"] = np.nan
        df2.loc[mask, "y"] = np.nan
    gdf = cudf.DataFrame(df)
    gdf2 = cudf.DataFrame(df2)
    if index:
        df = df.set_index(index)
        df2 = df2.set_index(index)
        gdf = gdf.set_index(index)
        gdf2 = gdf2.set_index(index)
    return df, df2, gdf, gdf2


@pytest.mark.parametrize("nulls", ["none", "some", "all"])
@pytest.mark.parametrize("index", [False, "z", "y"])
def test_concat_dataframe(index, nulls, axis_0):
    if index == "y" and nulls in ("some", "all"):
        pytest.skip("nulls in columns, dont index")
    df, df2, gdf, gdf2 = make_frames(index, nulls=nulls)
    # Make empty frame
    gdf_empty1 = gdf2[:0]
    assert len(gdf_empty1) == 0
    df_empty1 = gdf_empty1.to_pandas()

    # DataFrame
    res = cudf.concat([gdf, gdf2, gdf, gdf_empty1], axis=axis_0).to_pandas()
    sol = pd.concat([df, df2, df, df_empty1], axis=axis_0)
    assert_eq(
        res,
        sol,
        check_names=False,
        check_categorical=False,
        check_index_type=True,
    )

    # Series
    for c in [i for i in ("x", "y", "z") if i != index]:
        res = cudf.concat([gdf[c], gdf2[c], gdf[c]], axis=axis_0).to_pandas()
        sol = pd.concat([df[c], df2[c], df[c]], axis=axis_0)
        assert_eq(
            res,
            sol,
            check_names=False,
            check_categorical=False,
            check_index_type=True,
        )

    # Index
    res = cudf.concat([gdf.index, gdf2.index], axis=axis_0).to_pandas()
    sol = df.index.append(df2.index)
    assert_eq(res, sol, check_names=False, check_categorical=False)


@pytest.mark.parametrize(
    "values",
    [["foo", "bar"], [1.0, 2.0], pd.Series(["one", "two"], dtype="category")],
)
def test_concat_all_nulls(values):
    pa = pd.Series(values)
    pb = pd.Series([None], dtype=pa.dtype)
    ps = pd.concat([pa, pb])

    ga = cudf.Series(values)
    gb = cudf.Series([None])
    gs = cudf.concat([ga, gb])

    assert_eq(
        ps,
        gs,
        check_dtype=False,
        check_categorical=False,
        check_index_type=True,
    )


def test_concat_errors():
    df, _df2, gdf, gdf2 = make_frames()

    # No objs
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": []}),
        rfunc_args_and_kwargs=([], {"objs": []}),
    )

    # All None
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": [None, None]}),
        rfunc_args_and_kwargs=([], {"objs": [None, None]}),
    )

    # Mismatched types
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": [df, df.index, df.x]}),
        rfunc_args_and_kwargs=([], {"objs": [gdf, gdf.index, gdf.x]}),
    )

    # Unknown type
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": ["bar", "foo"]}),
        rfunc_args_and_kwargs=([], {"objs": ["bar", "foo"]}),
    )

    # Mismatched index dtypes
    gdf3 = gdf2.copy()
    del gdf3["z"]
    gdf4 = gdf2.set_index("z")

    with pytest.raises(ValueError, match="All columns must be the same type"):
        cudf.concat([gdf3, gdf4])

    # Bad axis value
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=(
            [],
            {"objs": [gdf.to_pandas(), gdf2.to_pandas()], "axis": "bad_value"},
        ),
        rfunc_args_and_kwargs=([], {"objs": [gdf, gdf2], "axis": "bad_value"}),
    )


def test_concat_misordered_columns():
    df, df2, gdf, gdf2 = make_frames(False)
    gdf2 = gdf2[["z", "x", "y"]]
    df2 = df2[["z", "x", "y"]]

    res = cudf.concat([gdf, gdf2]).to_pandas()
    sol = pd.concat([df, df2], sort=False)

    assert_eq(
        res,
        sol,
        check_names=False,
        check_categorical=False,
        check_index_type=True,
    )


def test_concat_columns(axis_1):
    rng = np.random.default_rng(seed=0)
    pdf1 = pd.DataFrame(rng.integers(10, size=(5, 3)), columns=[1, 2, 3])
    pdf2 = pd.DataFrame(rng.integers(10, size=(5, 4)), columns=[4, 5, 6, 7])
    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    expect = pd.concat([pdf1, pdf2], axis=axis_1)
    got = cudf.concat([gdf1, gdf2], axis=axis_1)

    assert_eq(expect, got, check_index_type=True)


def test_concat_multiindex_dataframe(axis):
    gdf = cudf.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg.iloc[:, :1]
    pdg2 = pdg.iloc[:, 1:]
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    expected = pd.concat([pdg1, pdg2], axis=axis)
    result = cudf.concat([gdg1, gdg2], axis=axis)
    assert_eq(
        expected,
        result,
        check_index_type=True,
    )


def test_concat_multiindex_series():
    gdf = cudf.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg["y"]
    pdg2 = pdg["z"]
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    assert_eq(
        cudf.concat([gdg1, gdg2]),
        pd.concat([pdg1, pdg2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1)
    )


def test_concat_multiindex_dataframe_and_series():
    gdf = cudf.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg[["y", "z"]]
    pdg2 = pdg["z"]
    pdg2.name = "a"
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    assert_eq(
        cudf.concat([gdg1, gdg2], axis=1),
        pd.concat([pdg1, pdg2], axis=1),
        check_index_type=True,
    )


def test_concat_multiindex_series_and_dataframe():
    gdf = cudf.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg["z"]
    pdg2 = pdg[["y", "z"]]
    pdg1.name = "a"
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    assert_eq(
        cudf.concat([gdg1, gdg2], axis=1),
        pd.concat([pdg1, pdg2], axis=1),
        check_index_type=True,
    )


@pytest.mark.parametrize("myindex", ["a", "b"])
def test_concat_string_index_name(myindex):
    # GH-Issue #3420
    data = {"a": [123, 456], "b": ["s1", "s2"]}
    df1 = cudf.DataFrame(data).set_index(myindex)
    df2 = df1.copy()
    df3 = cudf.concat([df1, df2])

    assert df3.index.name == myindex


def test_pandas_concat_compatibility_axis1():
    d1 = cudf.datasets.randomdata(
        3, dtypes={"a": float, "ind": float}
    ).set_index("ind")
    d2 = cudf.datasets.randomdata(
        3, dtypes={"b": float, "ind": float}
    ).set_index("ind")
    d3 = cudf.datasets.randomdata(
        3, dtypes={"c": float, "ind": float}
    ).set_index("ind")
    d4 = cudf.datasets.randomdata(
        3, dtypes={"d": float, "ind": float}
    ).set_index("ind")
    d5 = cudf.datasets.randomdata(
        3, dtypes={"e": float, "ind": float}
    ).set_index("ind")

    pd1 = d1.to_pandas()
    pd2 = d2.to_pandas()
    pd3 = d3.to_pandas()
    pd4 = d4.to_pandas()
    pd5 = d5.to_pandas()

    expect = pd.concat([pd1, pd2, pd3, pd4, pd5], axis=1)
    got = cudf.concat([d1, d2, d3, d4, d5], axis=1)

    assert_eq(
        got.sort_index(),
        expect.sort_index(),
        check_index_type=True,
    )


@pytest.mark.parametrize("index", [[0, 1, 2], [2, 1, 0], [5, 9, 10]])
@pytest.mark.parametrize("names", [False, (0, 1)])
@pytest.mark.parametrize(
    "data",
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", "c"], ["XX", "YY", "ZZ"]),
    ],
)
def test_pandas_concat_compatibility_axis1_overlap(index, names, data):
    s1 = cudf.Series(data[0], index=[0, 1, 2])
    s2 = cudf.Series(data[1], index=index)
    if names:
        s1.name = names[0]
        s2.name = names[1]
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = cudf.concat([s1, s2], axis=1)
    expect = pd.concat([ps1, ps2], axis=1)
    assert_eq(got, expect, check_index_type=True)


def test_pandas_concat_compatibility_axis1_eq_index():
    s1 = cudf.Series(["a", "b", "c"], index=[0, 1, 2])
    s2 = cudf.Series(["a", "b", "c"], index=[1, 1, 1])
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()

    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": [ps1, ps2], "axis": 1}),
        rfunc_args_and_kwargs=([], {"objs": [s1, s2], "axis": 1}),
    )


@pytest.mark.parametrize("name", [None, "a"])
def test_pandas_concat_compatibility_axis1_single_column(name):
    # Pandas renames series name `None` to 0
    # and preserves anything else
    s = cudf.Series([1, 2, 3], name=name)
    got = cudf.concat([s], axis=1)
    expected = pd.concat([s.to_pandas()], axis=1)
    assert_eq(expected, got)


def test_concat_duplicate_columns():
    cdf = cudf.DataFrame(
        {
            "id4": 4 * list(range(6)),
            "id5": 4 * list(reversed(range(6))),
            "v3": 6 * list(range(4)),
        }
    )
    cdf_std = cdf.groupby(["id4", "id5"])[["v3"]].std()
    cdf_med = cdf.groupby(["id4", "id5"])[["v3"]].quantile(q=0.5)
    with pytest.raises(NotImplementedError):
        cudf.concat([cdf_med, cdf_std], axis=1)


def test_concat_mixed_input():
    pdf1 = pd.DataFrame({"a": [10, 20, 30]})
    pdf2 = pd.DataFrame({"a": [11, 22, 33]})

    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    assert_eq(
        pd.concat([pdf1, None, pdf2, None]),
        cudf.concat([gdf1, None, gdf2, None]),
        check_index_type=True,
    )
    assert_eq(
        pd.concat([pdf1, None]),
        cudf.concat([gdf1, None]),
        check_index_type=True,
    )
    assert_eq(
        pd.concat([None, pdf2]),
        cudf.concat([None, gdf2]),
        check_index_type=True,
    )
    assert_eq(
        pd.concat([None, pdf2, pdf1]),
        cudf.concat([None, gdf2, gdf1]),
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [pd.Series([1, 2, 3]), pd.DataFrame({"a": [1, 2]})],
        [pd.Series([1, 2, 3]), pd.DataFrame({"a": []})],
        [pd.Series([], dtype="float64"), pd.DataFrame({"a": []})],
        [pd.Series([], dtype="float64"), pd.DataFrame({"a": [1, 2]})],
        [
            pd.Series([1, 2, 3.0, 1.2], name="abc"),
            pd.DataFrame({"a": [1, 2]}),
        ],
        [
            pd.Series(
                [1, 2, 3.0, 1.2], name="abc", index=[100, 110, 120, 130]
            ),
            pd.DataFrame({"a": [1, 2]}),
        ],
        [
            pd.Series(
                [1, 2, 3.0, 1.2], name="abc", index=["a", "b", "c", "d"]
            ),
            pd.DataFrame({"a": [1, 2]}, index=["a", "b"]),
        ],
        [
            pd.Series(
                [1, 2, 3.0, 1.2, 8, 100],
                name="New name",
                index=["a", "b", "c", "d", "e", "f"],
            ),
            pd.DataFrame(
                {"a": [1, 2, 4, 10, 11, 12]},
                index=["a", "b", "c", "d", "e", "f"],
            ),
        ],
        [
            pd.Series(
                [1, 2, 3.0, 1.2, 8, 100],
                name="New name",
                index=["a", "b", "c", "d", "e", "f"],
            ),
            pd.DataFrame(
                {"a": [1, 2, 4, 10, 11, 12]},
                index=["a", "b", "c", "d", "e", "f"],
            ),
        ]
        * 7,
    ],
)
def test_concat_series_dataframe_input(objs):
    pd_objs = objs
    gd_objs = [cudf.from_pandas(obj) for obj in objs]

    expected = pd.concat(pd_objs)
    actual = cudf.concat(gd_objs)

    assert_eq(
        expected.fillna(-1),
        actual.fillna(-1),
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.Series(["a", "b", "c", "d"]),
            pd.Series(["1", "2", "3", "4"]),
            pd.DataFrame({"first col": ["10", "11", "12", "13"]}),
        ],
        [
            pd.Series(["a", "b", "c", "d"]),
            pd.Series(["1", "2", "3", "4"]),
            pd.DataFrame(
                {
                    "first col": ["10", "11", "12", "13"],
                    "second col": ["a", "b", "c", "d"],
                }
            ),
        ],
        [
            pd.Series(["a", "b", "c"]),
            pd.Series(["1", "2", "3", "4"]),
            pd.DataFrame(
                {
                    "first col": ["10", "11", "12", "13"],
                    "second col": ["a", "b", "c", "d"],
                }
            ),
        ],
    ],
)
def test_concat_series_dataframe_input_str(objs):
    pd_objs = objs
    gd_objs = [cudf.from_pandas(obj) for obj in objs]

    expected = pd.concat(pd_objs)
    actual = cudf.concat(gd_objs)
    assert_eq(expected, actual, check_dtype=False, check_index_type=False)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame(
            {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
        ),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[200]),
        pd.DataFrame([], index=[100]),
        pd.DataFrame({"cat": pd.Series(["one", "two"], dtype="category")}),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        [
            pd.DataFrame(
                {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        ],
        [
            pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"l": [10]}),
            pd.DataFrame({"l": [10]}, index=[200]),
            pd.DataFrame(
                {"cat": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame(
                {"cat": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
    ],
)
def test_concat_empty_dataframes(df, other, ignore_index):
    other_pd = [df, *other]

    gdf = cudf.from_pandas(df)
    other_gd = [gdf] + [cudf.from_pandas(o) for o in other]

    expected = pd.concat(other_pd, ignore_index=ignore_index)
    actual = cudf.concat(other_gd, ignore_index=ignore_index)
    if expected.shape != df.shape:
        for key, col in actual[actual.columns].items():
            if isinstance(col.dtype, cudf.CategoricalDtype):
                if not isinstance(expected[key].dtype, pd.CategoricalDtype):
                    # TODO: Pandas bug:
                    # https://github.com/pandas-dev/pandas/issues/42840
                    expected[key] = expected[key].fillna("-1").astype("str")
                else:
                    expected[key] = (
                        expected[key]
                        .cat.add_categories(["-1"])
                        .fillna("-1")
                        .astype("str")
                    )
                actual[key] = col.astype("str").fillna("-1")
            else:
                expected[key] = expected[key].fillna(-1)
                actual[key] = col.fillna(-1)
        assert_eq(expected, actual, check_dtype=False, check_index_type=True)
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=False,
        )


@pytest.mark.parametrize(
    "data",
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", "c"], ["XX", "YY", "ZZ"]),
    ],
)
def test_concat_empty_and_nonempty_series(ignore_index, data, axis_0):
    s1 = cudf.Series()
    s2 = cudf.Series(data[0])
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = cudf.concat([s1, s2], axis=axis_0, ignore_index=ignore_index)
    expect = pd.concat([ps1, ps2], axis=axis_0, ignore_index=ignore_index)

    assert_eq(got, expect, check_index_type=True, check_dtype=False)


def test_concat_two_empty_series(ignore_index, axis_0):
    s1 = cudf.Series()
    s2 = cudf.Series()
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = cudf.concat([s1, s2], axis=axis_0, ignore_index=ignore_index)
    expect = pd.concat([ps1, ps2], axis=axis_0, ignore_index=ignore_index)

    assert_eq(got, expect, check_index_type=True)


@pytest.mark.parametrize(
    "key2", [[0, 1], [1, 0]], ids=["matching", "different"]
)
def test_concat_dataframe_with_multiindex(key2):
    gdf1 = cudf.DataFrame({"k1": [0, 1], "k2": [2, 3], "v1": [4, 5]})
    gdf1 = gdf1.set_index(["k1", "k2"])

    gdf2 = cudf.DataFrame({"k1": key2, "k2": [3, 2], "v2": [6, 7]})
    gdf2 = gdf2.set_index(["k1", "k2"])

    pdf1 = gdf1.to_pandas()
    pdf2 = gdf2.to_pandas()

    actual = cudf.concat([gdf1, gdf2], axis=1)
    expected = pd.concat([pdf1, pdf2], axis=1)

    # Will need to sort_index before comparing as
    # ordering is not deterministic in case of pandas
    # multiIndex with concat.
    assert_eq(
        expected.sort_index(),
        actual.sort_index(),
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))}
            ),
        ],
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))},
                index=["k", "l", "m", "n", "o", "p", "q", "r", "s", "t"],
            ),
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))},
                index=["a", "b", "c", "d", "z", "f", "g", "h", "i", "w"],
            ),
        ],
    ],
)
def test_concat_join(objs, ignore_index, sort, join, axis):
    axis = 0
    gpu_objs = [cudf.from_pandas(o) for o in objs]

    assert_eq(
        pd.concat(
            objs, sort=sort, join=join, ignore_index=ignore_index, axis=axis
        ),
        cudf.concat(
            gpu_objs,
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        ),
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))}
            ),
        ],
    ],
)
def test_concat_join_axis_1_dup_error(objs):
    gpu_objs = [cudf.from_pandas(o) for o in objs]
    # we do not support duplicate columns
    with pytest.raises(NotImplementedError):
        assert_eq(
            pd.concat(
                objs,
                axis=1,
            ),
            cudf.concat(
                gpu_objs,
                axis=1,
            ),
        )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(
                {"l": range(10, 20), "m": list(map(float, range(10, 20)))}
            ),
        ],
    ],
)
def test_concat_join_axis_1(objs, ignore_index, sort, join, axis):
    # no duplicate columns
    axis = 1
    gpu_objs = [cudf.from_pandas(o) for o in objs]
    expected = pd.concat(
        objs, sort=sort, join=join, ignore_index=ignore_index, axis=axis
    )
    actual = cudf.concat(
        gpu_objs,
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )

    assert_eq(expected, actual, check_index_type=True)


def test_concat_join_many_df_and_empty_df(ignore_index, sort, join, axis):
    # no duplicate columns
    pdf1 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    pdf2 = pd.DataFrame(
        {"l": range(10, 20), "m": list(map(float, range(10, 20)))}
    )
    pdf3 = pd.DataFrame({"j": [1, 2], "k": [1, 2], "s": [1, 2], "t": [1, 2]})
    pdf_empty1 = pd.DataFrame()

    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)
    gdf3 = cudf.from_pandas(pdf3)
    gdf_empty1 = cudf.from_pandas(pdf_empty1)

    assert_eq(
        pd.concat(
            [pdf1, pdf2, pdf3, pdf_empty1],
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        ),
        cudf.concat(
            [gdf1, gdf2, gdf3, gdf_empty1],
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        ),
        check_index_type=False,
    )


def test_concat_join_one_df(ignore_index, sort, join, axis):
    pdf1 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    gdf1 = cudf.from_pandas(pdf1)
    expected = pd.concat(
        [pdf1], sort=sort, join=join, ignore_index=ignore_index, axis=axis
    )
    actual = cudf.concat(
        [gdf1], sort=sort, join=join, ignore_index=ignore_index, axis=axis
    )

    assert_eq(expected, actual, check_index_type=True)


@pytest.mark.parametrize(
    "pdf1,pdf2",
    [
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]}),
        ),
        (
            pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6]}, index=["p", "q", "r"]
            ),
            pd.DataFrame(
                {"c": [7, 8, 9], "d": [10, 11, 12]}, index=["r", "p", "z"]
            ),
        ),
    ],
)
def test_concat_join_no_overlapping_columns(
    pdf1, pdf2, ignore_index, sort, join, axis
):
    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    expected = pd.concat(
        [pdf1, pdf2],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    actual = cudf.concat(
        [gdf1, gdf2],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )

    assert_eq(expected, actual, check_index_type=True)


def test_concat_join_no_overlapping_columns_many_and_empty(
    ignore_index, sort, join, axis
):
    pdf4 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf5 = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
    pdf6 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    pdf_empty = pd.DataFrame()

    gdf4 = cudf.from_pandas(pdf4)
    gdf5 = cudf.from_pandas(pdf5)
    gdf6 = cudf.from_pandas(pdf6)
    gdf_empty = cudf.from_pandas(pdf_empty)

    expected = pd.concat(
        [pdf4, pdf5, pdf6, pdf_empty],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    actual = cudf.concat(
        [gdf4, gdf5, gdf6, gdf_empty],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    assert_eq(
        expected,
        actual,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6]}, index=["z", "t", "k"]
            ),
            pd.DataFrame(
                {"c": [7, 8, 9], "d": [10, 11, 12]}, index=["z", "t", "k"]
            ),
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                index=["z", "t", "k", "a", "b", "c", "d", "e", "f", "g"],
            ),
            pd.DataFrame(index=pd.Index([], dtype="str")),
        ],
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]}),
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(index=pd.Index([], dtype="str")),
        ],
        [
            pd.DataFrame(
                {"a": [1, 2, 3], "nb": [10, 11, 12]}, index=["Q", "W", "R"]
            ),
            None,
        ],
    ],
)
def test_concat_join_no_overlapping_columns_many_and_empty2(
    objs, ignore_index, sort, join, axis
):
    objs_gd = [cudf.from_pandas(o) if o is not None else o for o in objs]

    expected = pd.concat(
        objs,
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    actual = cudf.concat(
        objs_gd,
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    assert_eq(expected, actual, check_index_type=False)


def test_concat_join_no_overlapping_columns_empty_df_basic(
    ignore_index, sort, join, axis
):
    pdf6 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    pdf_empty = pd.DataFrame()

    gdf6 = cudf.from_pandas(pdf6)
    gdf_empty = cudf.from_pandas(pdf_empty)

    expected = pd.concat(
        [pdf6, pdf_empty],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    actual = cudf.concat(
        [gdf6, gdf_empty],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    assert_eq(
        expected,
        actual,
        check_index_type=True,
        check_column_type=False,
    )


def test_concat_join_series(ignore_index, sort, join, axis):
    s1 = cudf.Series(["a", "b", "c"])
    s2 = cudf.Series(["a", "b"])
    s3 = cudf.Series(["a", "b", "c", "d"])
    s4 = cudf.Series(dtype="str")

    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    ps3 = s3.to_pandas()
    ps4 = s4.to_pandas()

    expected = pd.concat(
        [ps1, ps2, ps3, ps4],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    actual = cudf.concat(
        [s1, s2, s3, s4],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )

    assert_eq(
        expected,
        actual,
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame(
            {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
        ),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[200]),
        pd.DataFrame([], index=[100]),
        pd.DataFrame({"cat": pd.Series(["one", "two"], dtype="category")}),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        [
            pd.DataFrame(
                {"b": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        ],
        [
            pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"l": [10]}),
            pd.DataFrame({"k": [10]}, index=[200]),
            pd.DataFrame(
                {"cat": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame(
                {"cat": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
    ],
)
def test_concat_join_empty_dataframes(
    request, df, other, ignore_index, join, sort
):
    axis = 0
    other_pd = [df, *other]
    gdf = cudf.from_pandas(df)
    other_gd = [gdf] + [cudf.from_pandas(o) for o in other]

    expected = pd.concat(
        other_pd, ignore_index=ignore_index, axis=axis, join=join, sort=sort
    )
    actual = cudf.concat(
        other_gd, ignore_index=ignore_index, axis=axis, join=join, sort=sort
    )
    if (
        join == "outer"
        and any(
            isinstance(dtype, pd.CategoricalDtype)
            for dtype in df.dtypes.tolist()
        )
        and any(
            isinstance(dtype, pd.CategoricalDtype)
            for other_df in other
            for dtype in other_df.dtypes.tolist()
        )
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/42840"
            )
        )
    assert_eq(
        expected,
        actual,
        check_dtype=False,
        check_column_type=False,
    )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame(
            {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
        ),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"m": [10]}, index=[200]),
        pd.DataFrame([], index=[100]),
        pd.DataFrame({"cat": pd.Series(["one", "two"], dtype="category")}),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        [
            pd.DataFrame(
                {"b": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("CD")),
        ],
        [
            pd.DataFrame({"g": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"h": [10]}),
            pd.DataFrame({"k": [10]}, index=[200]),
            pd.DataFrame(
                {"dog": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame(
                {"bird": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
    ],
)
def test_concat_join_empty_dataframes_axis_1(
    df, other, ignore_index, axis, join, sort
):
    # no duplicate columns
    axis = 1
    other_pd = [df, *other]
    gdf = cudf.from_pandas(df)
    other_gd = [gdf] + [cudf.from_pandas(o) for o in other]

    expected = pd.concat(
        other_pd,
        ignore_index=ignore_index,
        axis=axis,
        join=join,
        sort=sort,
    )
    actual = cudf.concat(
        other_gd,
        ignore_index=ignore_index,
        axis=axis,
        join=join,
        sort=sort,
    )
    if expected.shape != df.shape:
        if axis == 0:
            for key, col in actual[actual.columns].items():
                if isinstance(expected[key].dtype, pd.CategoricalDtype):
                    expected[key] = expected[key].fillna("-1")
                    actual[key] = col.astype("str").fillna("-1")
            # if not expected.empty:
            assert_eq(
                expected.fillna(-1),
                actual.fillna(-1),
                check_dtype=False,
                check_index_type=False
                if len(expected) == 0 or actual.empty
                else True,
                check_column_type=False,
            )
        else:
            # no need to fill in if axis=1
            assert_eq(
                expected,
                actual,
                check_index_type=False,
                check_column_type=False,
            )
    assert_eq(
        expected, actual, check_index_type=False, check_column_type=False
    )


def test_concat_preserve_order():
    """Ensure that order is preserved on 'inner' concatenations."""
    df = pd.DataFrame([["d", 3, 4.0], ["c", 4, 5.0]], columns=["c", "b", "a"])
    dfs = [df, df]

    assert_eq(
        pd.concat(dfs, join="inner"),
        cudf.concat([cudf.DataFrame(df) for df in dfs], join="inner"),
        check_index_type=True,
    )


@pytest.mark.parametrize("typ", [cudf.DataFrame, cudf.Series])
def test_concat_single_object(ignore_index, typ):
    """Ensure that concat on a single object does not change it."""
    obj = typ([1, 2, 3])
    assert_eq(
        cudf.concat([obj], ignore_index=ignore_index, axis=0),
        obj,
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "ltype",
    [Decimal64Dtype(3, 1), Decimal64Dtype(7, 2), Decimal64Dtype(8, 4)],
)
@pytest.mark.parametrize(
    "rtype",
    [
        Decimal64Dtype(3, 2),
        Decimal64Dtype(8, 4),
        cudf.Decimal128Dtype(3, 2),
        cudf.Decimal32Dtype(8, 4),
    ],
)
def test_concat_decimal_dataframe(ltype, rtype):
    rng = np.random.default_rng(seed=0)
    gdf1 = cudf.DataFrame(
        {"id": rng.integers(0, 10, 3), "val": ["22.3", "59.5", "81.1"]}
    )
    gdf2 = cudf.DataFrame(
        {"id": rng.integers(0, 10, 3), "val": ["2.35", "5.59", "8.14"]}
    )

    gdf1["val"] = gdf1["val"].astype(ltype)
    gdf2["val"] = gdf2["val"].astype(rtype)

    pdf1 = gdf1.to_pandas()
    pdf2 = gdf2.to_pandas()

    got = cudf.concat([gdf1, gdf2])
    expected = pd.concat([pdf1, pdf2])

    assert_eq(expected, got, check_index_type=True)


@pytest.mark.parametrize("ltype", [Decimal64Dtype(4, 1), Decimal64Dtype(8, 2)])
@pytest.mark.parametrize(
    "rtype",
    [
        Decimal64Dtype(4, 3),
        Decimal64Dtype(10, 4),
        Decimal32Dtype(8, 3),
        Decimal128Dtype(18, 3),
    ],
)
def test_concat_decimal_series(ltype, rtype):
    gs1 = cudf.Series(["228.3", "559.5", "281.1"]).astype(ltype)
    gs2 = cudf.Series(["2.345", "5.259", "8.154"]).astype(rtype)

    ps1 = gs1.to_pandas()
    ps2 = gs2.to_pandas()

    got = cudf.concat([gs1, gs2])
    expected = pd.concat([ps1, ps2])

    assert_eq(expected, got, check_index_type=True)


@pytest.mark.parametrize(
    "data1, dtype1, index1, data2, dtype2, index2, data3, dtype3, index3, expected_data, expected_dtype, expected_index",
    [
        [
            {"val": [Decimal("42.5"), Decimal("8.7")]},
            Decimal64Dtype(5, 2),
            None,
            {"val": [Decimal("9.23"), Decimal("-67.49")]},
            Decimal64Dtype(6, 4),
            None,
            {"val": [8, -5]},
            "int32",
            None,
            {
                "val": [
                    Decimal("42.5"),
                    Decimal("8.7"),
                    Decimal("9.23"),
                    Decimal("-67.49"),
                    Decimal("8"),
                    Decimal("-5"),
                ]
            },
            Decimal32Dtype(7, 4),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            {"val": [Decimal("95.2"), Decimal("23.4")]},
            Decimal64Dtype(5, 2),
            None,
            {"val": [54, 509]},
            "uint16",
            None,
            {"val": [24, -48]},
            "int32",
            None,
            {
                "val": [
                    Decimal("95.2"),
                    Decimal("23.4"),
                    Decimal("54"),
                    Decimal("509"),
                    Decimal("24"),
                    Decimal("-48"),
                ]
            },
            Decimal32Dtype(5, 2),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            {"val": [Decimal("36.56"), Decimal("-59.24")]},
            Decimal64Dtype(9, 4),
            None,
            {"val": [403.21, 45.13]},
            "float32",
            None,
            {"val": [52.262, -49.25]},
            "float64",
            None,
            {
                "val": [
                    Decimal("36.56"),
                    Decimal("-59.24"),
                    Decimal("403.21"),
                    Decimal("45.13"),
                    Decimal("52.262"),
                    Decimal("-49.25"),
                ]
            },
            Decimal32Dtype(9, 4),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            {"val": [Decimal("9563.24"), Decimal("236.633")]},
            Decimal64Dtype(9, 4),
            None,
            {"val": [5393, -95832]},
            "int64",
            None,
            {"val": [-29.234, -31.945]},
            "float64",
            None,
            {
                "val": [
                    Decimal("9563.24"),
                    Decimal("236.633"),
                    Decimal("5393"),
                    Decimal("-95832"),
                    Decimal("-29.234"),
                    Decimal("-31.945"),
                ]
            },
            Decimal32Dtype(9, 4),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            {"val": [Decimal("95633.24"), Decimal("236.633")]},
            Decimal128Dtype(19, 4),
            None,
            {"val": [5393, -95832]},
            "int64",
            None,
            {"val": [-29.234, -31.945]},
            "float64",
            None,
            {
                "val": [
                    Decimal("95633.24"),
                    Decimal("236.633"),
                    Decimal("5393"),
                    Decimal("-95832"),
                    Decimal("-29.234"),
                    Decimal("-31.945"),
                ]
            },
            Decimal128Dtype(19, 4),
            [0, 1, 0, 1, 0, 1],
        ],
    ],
)
def test_concat_decimal_numeric_dataframe(
    data1,
    dtype1,
    index1,
    data2,
    dtype2,
    index2,
    data3,
    dtype3,
    index3,
    expected_data,
    expected_dtype,
    expected_index,
):
    df1 = cudf.DataFrame(data1, dtype=dtype1, index=index1)
    df2 = cudf.DataFrame(data2, dtype=dtype2, index=index2)
    df3 = cudf.DataFrame(data3, dtype=dtype3, index=index3)
    expected = cudf.DataFrame(
        expected_data, dtype=expected_dtype, index=expected_index
    )
    df = cudf.concat([df1, df2, df3])
    assert_eq(df, expected, check_index_type=True)
    assert_eq(df.val.dtype, expected.val.dtype)


@pytest.mark.parametrize(
    "data1, dtype1, index1, data2, dtype2, index2, data3, dtype3, index3, expected_data, expected_dtype, expected_index",
    [
        [
            [Decimal("32.8"), Decimal("-87.7")],
            Decimal64Dtype(6, 2),
            None,
            [Decimal("101.243"), Decimal("-92.449")],
            Decimal64Dtype(9, 6),
            None,
            [94, -22],
            "int32",
            None,
            [
                Decimal("32.8"),
                Decimal("-87.7"),
                Decimal("101.243"),
                Decimal("-92.449"),
                Decimal("94"),
                Decimal("-22"),
            ],
            Decimal64Dtype(10, 6),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            [Decimal("7.2"), Decimal("122.1")],
            Decimal64Dtype(5, 2),
            None,
            [33, 984],
            "uint32",
            None,
            [593, -702],
            "int32",
            None,
            [
                Decimal("7.2"),
                Decimal("122.1"),
                Decimal("33"),
                Decimal("984"),
                Decimal("593"),
                Decimal("-702"),
            ],
            Decimal32Dtype(5, 2),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            [Decimal("982.94"), Decimal("-493.626")],
            Decimal64Dtype(9, 4),
            None,
            [847.98, 254.442],
            "float32",
            None,
            [5299.262, -2049.25],
            "float64",
            None,
            [
                Decimal("982.94"),
                Decimal("-493.626"),
                Decimal("847.98"),
                Decimal("254.442"),
                Decimal("5299.262"),
                Decimal("-2049.25"),
            ],
            Decimal32Dtype(9, 4),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            [Decimal("492.204"), Decimal("-72824.455")],
            Decimal64Dtype(9, 4),
            None,
            [8438, -27462],
            "int64",
            None,
            [-40.292, 49202.953],
            "float64",
            None,
            [
                Decimal("492.204"),
                Decimal("-72824.455"),
                Decimal("8438"),
                Decimal("-27462"),
                Decimal("-40.292"),
                Decimal("49202.953"),
            ],
            Decimal32Dtype(9, 4),
            [0, 1, 0, 1, 0, 1],
        ],
        [
            [Decimal("492.204"), Decimal("-72824.455")],
            Decimal64Dtype(10, 4),
            None,
            [Decimal("8438"), Decimal("-27462")],
            Decimal32Dtype(9, 4),
            None,
            [Decimal("-40.292"), Decimal("49202.953")],
            Decimal128Dtype(19, 4),
            None,
            [
                Decimal("492.204"),
                Decimal("-72824.455"),
                Decimal("8438"),
                Decimal("-27462"),
                Decimal("-40.292"),
                Decimal("49202.953"),
            ],
            Decimal128Dtype(19, 4),
            [0, 1, 0, 1, 0, 1],
        ],
    ],
)
def test_concat_decimal_numeric_series(
    data1,
    dtype1,
    index1,
    data2,
    dtype2,
    index2,
    data3,
    dtype3,
    index3,
    expected_data,
    expected_dtype,
    expected_index,
):
    s1 = cudf.Series(data1, dtype=dtype1, index=index1)
    s2 = cudf.Series(data2, dtype=dtype2, index=index2)
    s3 = cudf.Series(data3, dtype=dtype3, index=index3)
    expected = cudf.Series(
        expected_data, dtype=expected_dtype, index=expected_index
    )
    s = cudf.concat([s1, s2, s3])
    assert_eq(s, expected, check_index_type=True)


@pytest.mark.parametrize(
    "data1, dtype1, index1, data2, dtype2, index2, expected_data, expected_dtype, expected_index",
    [
        [
            [Decimal("955.22"), Decimal("8.2")],
            Decimal64Dtype(5, 2),
            None,
            ["2007-06-12", "2006-03-14"],
            "datetime64[s]",
            None,
            [
                "955.22",
                "8.20",
                "2007-06-12 00:00:00",
                "2006-03-14 00:00:00",
            ],
            None,
            [0, 1, 0, 1],
        ],
        [
            [Decimal("-52.44"), Decimal("365.22")],
            Decimal64Dtype(5, 2),
            None,
            np.arange(
                "2005-02-01T12", "2005-02-01T15", dtype="datetime64[h]"
            ).astype("datetime64[s]"),
            "datetime64[s]",
            None,
            [
                "-52.44",
                "365.22",
                "2005-02-01 12:00:00",
                "2005-02-01 13:00:00",
                "2005-02-01 14:00:00",
            ],
            None,
            [0, 1, 0, 1, 2],
        ],
        [
            [Decimal("753.0"), Decimal("94.22")],
            Decimal64Dtype(5, 2),
            None,
            [np.timedelta64(111, "s"), np.timedelta64(509, "s")],
            None,
            None,
            [
                "753.00",
                "94.22",
                "0 days 00:01:51",
                "0 days 00:08:29",
            ],
            None,
            [0, 1, 0, 1],
        ],
        [
            [Decimal("753.0"), Decimal("94.22")],
            Decimal64Dtype(5, 2),
            None,
            [np.timedelta64(940252, "s"), np.timedelta64(758385, "s")],
            None,
            None,
            [
                "753.00",
                "94.22",
                "10 days 21:10:52",
                "8 days 18:39:45",
            ],
            None,
            [0, 1, 0, 1],
        ],
    ],
)
def test_concat_decimal_non_numeric(
    data1,
    dtype1,
    index1,
    data2,
    dtype2,
    index2,
    expected_data,
    expected_dtype,
    expected_index,
):
    s1 = cudf.Series(data1, dtype=dtype1, index=index1)
    s2 = cudf.Series(data2, dtype=dtype2, index=index2)
    expected = cudf.Series(
        expected_data, dtype=expected_dtype, index=expected_index
    )
    s = cudf.concat([s1, s2])
    assert_eq(s, expected, check_index_type=True)


def test_concat_struct_column():
    s1 = cudf.Series([{"a": 5}, {"c": "hello"}, {"b": 7}])
    s2 = cudf.Series([{"a": 5, "c": "hello", "b": 7}])
    result = cudf.concat([s1, s2])
    # Field order depends on pyarrow's type inference (alphabetical
    # before pyarrow 23, insertion-order after), so derive the
    # expected output from s1's inferred dtype rather than hardcoding.
    # See https://github.com/apache/arrow/pull/48813
    fields = list(s1.dtype.fields)
    expected_data = [
        {f: (5 if f == "a" else None) for f in fields},
        {f: ("hello" if f == "c" else None) for f in fields},
        {f: (7 if f == "b" else None) for f in fields},
        {f: (5 if f == "a" else "hello" if f == "c" else 7) for f in fields},
    ]
    expected = cudf.Series(expected_data, index=[0, 1, 2, 0])
    assert_eq(result, expected, check_index_type=True)


@pytest.mark.parametrize(
    "frame1_cls, frame1_data, frame2_cls, frame2_data, expected_cls, expected_data",
    [
        (
            cudf.Series,
            {"data": [[{"b": 0}], [{"b": 1}], [{"b": 3}]]},
            cudf.Series,
            {"data": [[{"b": 10}], [{"b": 12}], None]},
            cudf.Series,
            {
                "data": [
                    [{"b": 0}],
                    [{"b": 1}],
                    [{"b": 3}],
                    [{"b": 10}],
                    [{"b": 12}],
                    None,
                ],
                "index": [0, 1, 2, 0, 1, 2],
            },
        ),
        (
            cudf.DataFrame,
            {"data": {"a": [[{"b": 0}], [{"b": 1}], [{"b": 3}]]}},
            cudf.DataFrame,
            {"data": {"a": [[{"b": 10}], [{"b": 12}], None]}},
            cudf.DataFrame,
            {
                "data": {
                    "a": [
                        [{"b": 0}],
                        [{"b": 1}],
                        [{"b": 3}],
                        [{"b": 10}],
                        [{"b": 12}],
                        None,
                    ]
                },
                "index": [0, 1, 2, 0, 1, 2],
            },
        ),
    ],
)
def test_concat_list_column(
    frame1_cls,
    frame1_data,
    frame2_cls,
    frame2_data,
    expected_cls,
    expected_data,
):
    frame1 = frame1_cls(**frame1_data)
    frame2 = frame2_cls(**frame2_data)
    expected = expected_cls(**expected_data)
    actual = cudf.concat([frame1, frame2])
    assert_eq(actual, expected, check_index_type=True)


def test_concat_categorical_ordering():
    # https://github.com/rapidsai/cudf/issues/11486
    sr = pd.Series(
        ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"], dtype="category"
    )
    sr = sr.cat.set_categories(["d", "a", "b", "c", "e"])

    df = pd.DataFrame({"a": sr})
    gdf = cudf.from_pandas(df)

    expect = pd.concat([df, df, df])
    got = cudf.concat([gdf, gdf, gdf])

    assert_eq(expect, got)


@pytest.fixture(params=["rangeindex", "index"])
def singleton_concat_index(request):
    if request.param == "rangeindex":
        return pd.RangeIndex(0, 4)
    else:
        return pd.Index(["a", "h", "g", "f"])


@pytest.fixture(params=["dataframe", "series"])
def singleton_concat_obj(request, singleton_concat_index):
    if request.param == "dataframe":
        return pd.DataFrame(
            {
                "b": [1, 2, 3, 4],
                "d": [7, 8, 9, 10],
                "a": [4, 5, 6, 7],
                "c": [10, 11, 12, 13],
            },
            index=singleton_concat_index,
        )
    else:
        return pd.Series([4, 5, 5, 6], index=singleton_concat_index)


def test_concat_singleton_sorting(
    axis, sort, ignore_index, singleton_concat_obj
):
    gobj = cudf.from_pandas(singleton_concat_obj)
    gconcat = cudf.concat(
        [gobj], axis=axis, sort=sort, ignore_index=ignore_index
    )
    pconcat = pd.concat(
        [singleton_concat_obj], axis=axis, sort=sort, ignore_index=ignore_index
    )
    assert_eq(pconcat, gconcat)


@pytest.mark.parametrize("axis", [2, "invalid"])
def test_concat_invalid_axis(axis):
    s = cudf.Series([1, 2, 3])
    with pytest.raises(ValueError):
        cudf.concat([s], axis=axis)


@pytest.mark.parametrize(
    "s1,s2",
    [
        ([1, 2], [[1, 2], [3, 4]]),
    ],
)
def test_concat_mixed_list_types_error(s1, s2):
    s1, s2 = cudf.Series(s1), cudf.Series(s2)

    with pytest.raises(NotImplementedError):
        cudf.concat([s1, s2], ignore_index=True)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(
            0,
            marks=pytest.mark.xfail(
                reason="concat dictionaries with axis=0 not implemented"
            ),
        ),
        1,
        "columns",
    ],
)
@pytest.mark.parametrize(
    "d",
    [
        {"first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}})},
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.DataFrame, {"data": {"A": [5, 6], "B": [7, 8]}}),
            "third": (cudf.DataFrame, {"data": {"C": [1, 2, 3]}}),
        },
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.DataFrame, {"data": {"C": [1, 2, 3]}}),
            "third": (cudf.DataFrame, {"data": {"A": [5, 6], "B": [7, 8]}}),
        },
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.DataFrame, {"data": {"C": [1, 2, 3]}}),
            "third": (cudf.DataFrame, {"data": {"A": [5, 6], "C": [7, 8]}}),
            "fourth": (cudf.DataFrame, {"data": {"B": [9, 10]}}),
        },
        pytest.param(
            {
                "first": (cudf.DataFrame, {"data": {2.0: [1, 1]}}),
                "second": (cudf.DataFrame, {"data": {"test": ["abc", "def"]}}),
            },
            marks=pytest.mark.xfail(
                reason=(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            ),
        ),
        {
            "first": (cudf.Series, {"data": [1, 2, 3]}),
            "second": (cudf.Series, {"data": [4, 5, 6]}),
        },
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.Series, {"data": [5, 6], "name": "C"}),
        },
        pytest.param(
            {
                "first": (
                    cudf.DataFrame,
                    {"data": {("A", "B"): [1, 2], "C": [3, 4]}},
                ),
                "second": (
                    cudf.DataFrame,
                    {"data": {"D": [5, 6], ("A", "B"): [7, 8]}},
                ),
            },
            marks=pytest.mark.xfail(
                reason=(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            ),
        ),
        pytest.param(
            {
                "first": (
                    cudf.DataFrame,
                    {"data": {("A", "B"): [3, 4], 2.0: [1, 1]}},
                ),
                "second": (
                    cudf.DataFrame,
                    {"data": {("C", "D"): [3, 4], 3.0: [5, 6]}},
                ),
            },
            marks=pytest.mark.xfail(
                reason=(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            ),
        ),
        {
            "first": (
                cudf.DataFrame,
                {"data": {(1, 2): [1, 2], (3, 4): [3, 4]}},
            ),
            "second": (
                cudf.DataFrame,
                {"data": {(1, 2): [5, 6], (5, 6): [7, 8]}},
            ),
        },
    ],
)
def test_concat_dictionary(d, axis):
    _dict = {k: c(**v) for k, (c, v) in d.items()}
    result = cudf.concat(_dict, axis=axis)
    expected = cudf.from_pandas(
        pd.concat({k: df.to_pandas() for k, df in _dict.items()}, axis=axis)
    )
    assert_eq(expected, result)


@pytest.mark.parametrize(
    "idx_cls, idx_data",
    [
        [cudf.Index, {"data": [1, 2, 3]}],
        [
            cudf.MultiIndex,
            {
                "levels": [[1, 2], ["blue", "red"]],
                "codes": [[0, 0, 1, 1], [1, 0, 1, 0]],
            },
        ],
        [cudf.CategoricalIndex, {"data": [1, 2, 3]}],
    ],
)
def test_concat_dict_incorrect_type_index(idx_cls, idx_data):
    idx = idx_cls(**idx_data)
    with pytest.raises(
        TypeError,
        match="cannot concatenate a dictionary containing indices",
    ):
        cudf.concat({"first": idx}, axis=1)


@pytest.mark.parametrize(
    "data",
    [
        [1, -2, 3, -4],
        [1.0, 12.221, 12.34, 13.324, 324.3242],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [-10, 11, -12, 13],
        [0.1, 0.002, 324.2332, 0.2342],
    ],
)
def test_series_concat_basic(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "123"],
        ["a"],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        ["abc", "123"],
        ["a"],
        ["+", "-", "!", "_", "="],
    ],
)
def test_series_concat_basic_str(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(["abc", "123"], index=[10, 20]),
        pd.Series(["a"], index=[10]),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        pd.Series(["abc", "123"], index=[50, 20]),
        pd.Series(["a"], index=[11]),
        pd.Series(["+", "-", "!", "_", "="], index=[12, 13, 14, 15, 16]),
    ],
)
def test_series_concat_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = cudf.from_pandas(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


def test_series_concat_error_mixed_types():
    gsr = cudf.Series([1, 2, 3, 4])
    other = cudf.Series(["a", "b", "c", "d"])

    with pytest.raises(
        TypeError,
        match=r"cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        cudf.concat([gsr, other])

    with pytest.raises(
        TypeError,
        match=r"cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        cudf.concat([gsr, gsr, other, gsr, other])


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([-1, 2, -3, 4], index=["a", "b", "c", "d"]),
        pd.Series(
            [1.0, 12.221, 12.34, 13.324, 324.3242],
            index=[
                "float one",
                "float two",
                "float three",
                "float four",
                "float five",
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [
            pd.Series([-10, 11, 12, 13], index=["a", "b", "c", "d"]),
            pd.Series([12, 14, -15, 27], index=["d", "e", "z", "x"]),
        ],
        [
            pd.Series(
                [0.1, 0.002, 324.2332, 0.2342], index=["-", "+", "%", "#"]
            ),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ]
        * 3,
    ],
)
def test_series_concat_list_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = [cudf.from_pandas(obj) for obj in others]

    expected = pd.concat([psr, *other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, *other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


def test_series_concat_existing_buffers():
    a1 = np.arange(10, dtype=np.float64)
    gs = cudf.Series(a1)

    # Add new buffer
    a2 = cudf.Series(np.arange(5))
    gs = cudf.concat([gs, a2])
    assert len(gs) == 15
    np.testing.assert_equal(gs.to_numpy(), np.hstack([a1, a2.to_numpy()]))

    # Ensure appending to previous buffer
    a3 = cudf.Series(np.arange(3))
    gs = cudf.concat([gs, a3])
    assert len(gs) == 18
    a4 = np.hstack([a1, a2.to_numpy(), a3.to_numpy()])
    np.testing.assert_equal(gs.to_numpy(), a4)

    # Appending different dtype
    a5 = cudf.Series(np.array([1, 2, 3], dtype=np.int32))
    a6 = cudf.Series(np.array([4.5, 5.5, 6.5], dtype=np.float64))
    gs = cudf.concat([a5, a6])
    np.testing.assert_equal(
        gs.to_numpy(), np.hstack([a5.to_numpy(), a6.to_numpy()])
    )
    gs = cudf.concat([cudf.Series(a6), a5])
    np.testing.assert_equal(
        gs.to_numpy(), np.hstack([a6.to_numpy(), a5.to_numpy()])
    )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [[1, 2], [10, 100]],
        [[1, 2, 10, 100, 0.1, 0.2, 0.0021]],
        [[]],
        [[], [], [], []],
        [[0.23, 0.00023, -10.00, 100, 200, 1000232, 1232.32323]],
    ],
)
def test_dataframe_concat_lists(df, other, sort, ignore_index):
    pdf = df
    other_pd = [pd.DataFrame(o) for o in other]

    gdf = cudf.from_pandas(df)
    other_gd = [cudf.from_pandas(o) for o in other_pd]

    expected = pd.concat(
        [pdf, *other_pd], sort=sort, ignore_index=ignore_index
    )
    actual = cudf.concat(
        [gdf, *other_gd], sort=sort, ignore_index=ignore_index
    )

    if expected.shape != df.shape:
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=not gdf.empty,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=len(gdf.columns) != 0,
        )


def test_dataframe_concat_series_without_name():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    pdf = df.to_pandas()
    gs = cudf.Series([1, 2, 3])
    ps = gs.to_pandas()

    assert_eq(pd.concat([pdf, ps]), cudf.concat([df, gs]))


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [pd.DataFrame([[5, 6], [7, 8]], columns=list("AB"))],
        [
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("BD")),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("DE")),
        ],
        [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        [
            pd.DataFrame(
                {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        ],
        [
            pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"l": [10]}),
            pd.DataFrame({"l": [10]}, index=[200]),
        ],
        [pd.DataFrame([]), pd.DataFrame([], index=[100])],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                }
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        ],
    ],
)
def test_dataframe_concat_dataframe_lists(df, other, sort, ignore_index):
    pdf = df
    other_pd = other

    gdf = cudf.from_pandas(df)
    other_gd = [cudf.from_pandas(o) for o in other]

    expected = pd.concat(
        [pdf, *other_pd], sort=sort, ignore_index=ignore_index
    )
    actual = cudf.concat(
        [gdf, *other_gd], sort=sort, ignore_index=ignore_index
    )

    # In some cases, Pandas creates an empty Index([], dtype="object") for
    # columns whereas cudf creates a RangeIndex(0, 0).
    check_column_type = (
        False if len(expected.columns) == len(df.columns) == 0 else True
    )

    if expected.shape != df.shape:
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=check_column_type,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=check_column_type,
        )


def test_dataframe_concat_series_mixed_index():
    df = cudf.DataFrame({"first": [], "d": []})
    pdf = df.to_pandas(arrow_type=True)
    df = cudf.from_pandas(pdf)

    sr = cudf.Series([1, 2, 3, 4])
    psr = sr.to_pandas(arrow_type=True)
    sr = cudf.from_pandas(psr)

    assert_eq(
        cudf.concat([df, sr], ignore_index=True),
        pd.concat([pdf, psr], ignore_index=True),
    )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({12: [], 22: []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=[0, 1], index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=[1, 0], index=[7, 8]),
        pd.DataFrame(
            {
                23: [315.3324, 3243.32432, 3232.332, -100.32],
                33: [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                0: [315.3324, 3243.32432, 3232.332, -100.32],
                1: [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.Series([10, 11, 23, 234, 13]),
        pd.Series([10, 11, 23, 234, 13], index=[11, 12, 13, 44, 33]),
        {1: 1},
        {0: 10, 1: 100, 2: 102},
    ],
)
def test_dataframe_concat_series(df, other, sort):
    pdf = df
    gdf = cudf.from_pandas(df)

    if isinstance(other, dict):
        other_pd = pd.Series(other)
    else:
        other_pd = other
    other_gd = cudf.from_pandas(other_pd)

    expected = pd.concat([pdf, other_pd], ignore_index=True, sort=sort)
    actual = cudf.concat([gdf, other_gd], ignore_index=True, sort=sort)

    if expected.shape != df.shape:
        # Ignore the column type comparison because pandas incorrectly
        # returns pd.Index([1, 2, 3], dtype="object") instead
        # of pd.Index([1, 2, 3], dtype="int64")
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=False,
            check_index_type=True,
        )
    else:
        assert_eq(expected, actual, check_index_type=not gdf.empty)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("BD")),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("DE")),
        pd.DataFrame(),
        pd.DataFrame(
            {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
        ),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[200]),
        pd.DataFrame([]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([], index=[100]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[0, 100, 200, 300],
        ),
    ],
)
def test_dataframe_concat_dataframe(df, other, sort, ignore_index):
    pdf = df
    other_pd = other

    gdf = cudf.from_pandas(df)
    other_gd = cudf.from_pandas(other)

    expected = pd.concat([pdf, other_pd], sort=sort, ignore_index=ignore_index)
    actual = cudf.concat([gdf, other_gd], sort=sort, ignore_index=ignore_index)

    # In empty dataframe cases, Pandas & cudf differ in columns
    # creation, pandas creates RangeIndex(0, 0)
    # whereas cudf creates an empty Index([], dtype="object").
    check_column_type = (
        False if len(expected.columns) == len(df.columns) == 0 else True
    )

    if expected.shape != df.shape:
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=check_column_type,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=check_column_type,
        )


@pytest.mark.parametrize("df_1_data", [{"a": [1, 2], "b": [1, 3]}, {}])
@pytest.mark.parametrize("df_2_data", [{"a": [], "b": []}, {}])
def test_concat_empty_dataframe(df_1_data, df_2_data):
    df_1 = cudf.DataFrame(df_1_data)
    df_2 = cudf.DataFrame(df_2_data)
    got = cudf.concat([df_1, df_2])
    expect = pd.concat([df_1.to_pandas(), df_2.to_pandas()], sort=False)

    # ignoring dtypes as pandas upcasts int to float
    # on concatenation with empty dataframes

    assert_eq(got, expect, check_dtype=False, check_index_type=True)


@pytest.mark.parametrize(
    "df1_d",
    [
        {"a": [1, 2], "b": [1, 2], "c": ["s1", "s2"], "d": [1.0, 2.0]},
        {"b": [1.9, 10.9], "c": ["s1", "s2"]},
        {"c": ["s1"], "b": pd.Series([None], dtype="float"), "a": [False]},
    ],
)
@pytest.mark.parametrize(
    "df2_d",
    [
        {"a": [1, 2, 3]},
        {"a": [1, None, 3], "b": [True, True, False], "c": ["s3", None, "s4"]},
        {"a": [], "b": []},
        {},
    ],
)
def test_concat_different_column_dataframe(request, df1_d, df2_d):
    pdf1 = pd.DataFrame(df1_d)
    pdf2 = pd.DataFrame(df2_d)

    got = cudf.concat(
        [
            cudf.from_pandas(pdf1),
            cudf.from_pandas(pdf2),
            cudf.from_pandas(pdf1),
        ],
        sort=False,
    )

    expect = pd.concat([pdf1, pdf2, pdf1], sort=False)
    xfail_pair = df2_d == {
        "a": [1, None, 3],
        "b": [True, True, False],
        "c": ["s3", None, "s4"],
    } and isinstance(df1_d["b"], pd.Series)
    request.applymarker(
        pytest.mark.xfail(
            xfail_pair,
            reason="As of pandas 3.0, pandas coerces to float, cuDF coerces to bool",
        )
    )

    assert_eq(got, expect, check_dtype=False, check_index_type=True)


@pytest.mark.parametrize(
    "ser_1", [pd.Series([1, 2, 3]), pd.Series([], dtype="float64")]
)
def test_concat_empty_series(ser_1):
    ser_2 = pd.Series([], dtype="float64")
    got = cudf.concat([cudf.Series(ser_1), cudf.Series(ser_2)])
    expect = pd.concat([ser_1, ser_2])

    assert_eq(got, expect, check_index_type=True)


def test_dataframe_concat_different_numerical_columns(
    numeric_and_temporal_types_as_str, numeric_and_temporal_types_as_str2
):
    df1 = pd.DataFrame(
        dict(
            x=pd.Series(np.arange(5)).astype(numeric_and_temporal_types_as_str)
        )
    )
    df2 = pd.DataFrame(
        dict(
            x=pd.Series(np.arange(5)).astype(
                numeric_and_temporal_types_as_str2
            )
        )
    )
    if (
        numeric_and_temporal_types_as_str != numeric_and_temporal_types_as_str2
        and "datetime" in numeric_and_temporal_types_as_str
        or "datetime" in numeric_and_temporal_types_as_str2
    ):
        with pytest.raises(TypeError):
            cudf.concat([df1, df2])
    else:
        pres = pd.concat([df1, df2])
        gres = cudf.concat([cudf.from_pandas(df1), cudf.from_pandas(df2)])
        assert_eq(pres, gres, check_dtype=False, check_index_type=True)


def test_dataframe_concat_different_column_types():
    df1 = cudf.Series([42], dtype=np.float64)
    df2 = cudf.Series(["a"], dtype="category")
    with pytest.raises(ValueError):
        cudf.concat([df1, df2])

    df2 = cudf.Series(["a string"])
    with pytest.raises(TypeError):
        cudf.concat([df1, df2])


@pytest.mark.parametrize("a", [[1, 2, 3], [1, 10, 30]])
@pytest.mark.parametrize("b", [[4, 5, 6], [-11, -100, 30]])
def test_concat_index(a, b):
    ser_a = pd.Series(a)
    ser_b = pd.Series(b)

    gser_a = cudf.Series(a)
    gser_b = cudf.Series(b)

    expected = pd.concat([ser_a, ser_b])
    actual = cudf.concat([gser_a, gser_b])

    assert len(expected) == len(actual)
    assert_eq(expected.index, actual.index)

    expected = pd.concat([ser_a, ser_b], ignore_index=True)
    actual = cudf.concat([gser_a, gser_b], ignore_index=True)

    assert len(expected) == len(actual)
    assert_eq(expected.index, actual.index)


def test_concat_with_axis():
    df1 = pd.DataFrame(dict(x=np.arange(5), y=np.arange(5)))
    df2 = pd.DataFrame(dict(a=np.arange(5), b=np.arange(5)))

    concat_df = pd.concat([df1, df2], axis=1)
    cdf1 = cudf.from_pandas(df1)
    cdf2 = cudf.from_pandas(df2)

    # concat only dataframes
    concat_cdf = cudf.concat([cdf1, cdf2], axis=1)
    assert_eq(concat_cdf, concat_df, check_index_type=True)

    # concat only series
    concat_s = pd.concat([df1.x, df1.y], axis=1)
    cs1 = cudf.Series(df1.x)
    cs2 = cudf.Series(df1.y)
    concat_cdf_s = cudf.concat([cs1, cs2], axis=1)

    assert_eq(concat_cdf_s, concat_s, check_index_type=True)

    rng = np.random.default_rng(seed=0)
    # concat series and dataframes
    s3 = pd.Series(rng.random(5))
    cs3 = cudf.Series(s3)

    concat_cdf_all = cudf.concat([cdf1, cs3, cdf2], axis=1)
    concat_df_all = pd.concat([df1, s3, df2], axis=1)
    assert_eq(concat_cdf_all, concat_df_all, check_index_type=True)

    # concat manual multi index
    midf1 = cudf.from_pandas(df1)
    midf1.index = cudf.MultiIndex(
        levels=[[0, 1, 2, 3], [0, 1]], codes=[[0, 1, 2, 3, 2], [0, 1, 0, 1, 0]]
    )
    midf2 = midf1[2:]
    midf2.index = cudf.MultiIndex(
        levels=[[3, 4, 5], [2, 0]], codes=[[0, 1, 2], [1, 0, 1]]
    )
    mipdf1 = midf1.to_pandas()
    mipdf2 = midf2.to_pandas()

    assert_eq(
        cudf.concat([midf1, midf2]),
        pd.concat([mipdf1, mipdf2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([midf2, midf1]),
        pd.concat([mipdf2, mipdf1]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([midf1, midf2, midf1]),
        pd.concat([mipdf1, mipdf2, mipdf1]),
        check_index_type=True,
    )

    rng = np.random.default_rng(seed=0)
    # concat groupby multi index
    gdf1 = cudf.DataFrame(
        {
            "x": rng.integers(0, 10, 10),
            "y": rng.integers(0, 10, 10),
            "z": rng.integers(0, 10, 10),
            "v": rng.integers(0, 10, 10),
        }
    )
    gdf2 = gdf1[5:]
    gdg1 = gdf1.groupby(["x", "y"]).min()
    gdg2 = gdf2.groupby(["x", "y"]).min()
    pdg1 = gdg1.to_pandas()
    pdg2 = gdg2.to_pandas()

    assert_eq(
        cudf.concat([gdg1, gdg2]),
        pd.concat([pdg1, pdg2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdg2, gdg1]),
        pd.concat([pdg2, pdg1]),
        check_index_type=True,
    )

    # series multi index concat
    gdgz1 = gdg1.z
    gdgz2 = gdg2.z
    pdgz1 = gdgz1.to_pandas()
    pdgz2 = gdgz2.to_pandas()

    assert_eq(
        cudf.concat([gdgz1, gdgz2]),
        pd.concat([pdgz1, pdgz2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdgz2, gdgz1]),
        pd.concat([pdgz2, pdgz1]),
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [pd.RangeIndex(0, 10), pd.RangeIndex(10, 20)],
        [pd.RangeIndex(10, 20), pd.RangeIndex(22, 40), pd.RangeIndex(50, 60)],
        [pd.RangeIndex(10, 20, 2), pd.RangeIndex(20, 40, 2)],
    ],
)
def test_range_index_concat(objs):
    cudf_objs = [cudf.from_pandas(obj) for obj in objs]

    actual = cudf.concat(cudf_objs)

    expected = objs[0]
    for obj in objs[1:]:
        expected = expected.append(obj)
    assert_eq(expected, actual)
