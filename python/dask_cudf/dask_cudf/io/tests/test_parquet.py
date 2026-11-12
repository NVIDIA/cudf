# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import math
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import dask
from dask import dataframe as dd
from dask.utils import natural_sort_key

import cudf

import dask_cudf
from dask_cudf._legacy.io.parquet import create_metadata_file

# Check if create_metadata_file is supported by
# the current dask.dataframe version
need_create_meta = pytest.mark.skipif(
    create_metadata_file is None,
    reason="Need create_metadata_file support in dask.dataframe.",
)

nrows = 40
npartitions = 15
df = pd.DataFrame(
    {
        "x": [i * 7 % 5 for i in range(nrows)],  # Not sorted
        "y": [i * 2.5 for i in range(nrows)],
    },
    index=pd.Index(range(nrows), name="index"),
)  # Sorted
ddf = dd.from_pandas(df, npartitions=npartitions)


def test_roundtrip_backend_dispatch(tmpdir):
    # Test ddf.read_parquet cudf-backend dispatch
    tmpdir = str(tmpdir)
    ddf.to_parquet(tmpdir, engine="pyarrow")
    with dask.config.set({"dataframe.backend": "cudf"}):
        ddf2 = dd.read_parquet(tmpdir, index=False, blocksize=None)
    assert isinstance(ddf2, dask_cudf.DataFrame)
    dd.assert_eq(ddf.reset_index(drop=False), ddf2)


@pytest.mark.parametrize("write_metadata_file", [True, False])
@pytest.mark.parametrize("divisions", [True, False])
def test_roundtrip_from_dask(tmpdir, divisions, write_metadata_file):
    tmpdir = str(tmpdir)
    ddf.to_parquet(
        tmpdir, write_metadata_file=write_metadata_file, engine="pyarrow"
    )
    files = sorted(
        (os.path.join(tmpdir, f) for f in os.listdir(tmpdir)),
        key=natural_sort_key,
    )

    # Read list of parquet files
    ddf2 = dask_cudf.read_parquet(files, calculate_divisions=divisions)
    dd.assert_eq(ddf, ddf2, check_divisions=divisions)

    # Specify columns=['x']
    ddf2 = dask_cudf.read_parquet(
        files, columns=["x"], calculate_divisions=divisions
    )
    dd.assert_eq(ddf[["x"]], ddf2, check_divisions=divisions)

    # Specify columns='y'
    ddf2 = dask_cudf.read_parquet(
        files, columns="y", calculate_divisions=divisions
    )
    dd.assert_eq(ddf["y"], ddf2, check_divisions=divisions)

    # Now include metadata
    ddf2 = dask_cudf.read_parquet(tmpdir, calculate_divisions=divisions)
    dd.assert_eq(ddf, ddf2, check_divisions=divisions)

    # Specify columns=['x'] (with metadata)
    ddf2 = dask_cudf.read_parquet(
        tmpdir, columns=["x"], calculate_divisions=divisions
    )
    dd.assert_eq(ddf[["x"]], ddf2, check_divisions=divisions)

    # Specify columns='y' (with metadata)
    ddf2 = dask_cudf.read_parquet(
        tmpdir, columns="y", calculate_divisions=divisions
    )
    dd.assert_eq(ddf["y"], ddf2, check_divisions=divisions)


def test_roundtrip_from_dask_index_false(tmpdir):
    tmpdir = str(tmpdir)
    ddf.to_parquet(tmpdir, engine="pyarrow")

    ddf2 = dask_cudf.read_parquet(tmpdir, index=False, blocksize=None)
    dd.assert_eq(ddf.reset_index(drop=False), ddf2)


def test_roundtrip_from_dask_none_index_false(tmpdir):
    tmpdir = str(tmpdir)
    path = os.path.join(tmpdir, "test.parquet")

    df2 = ddf.reset_index(drop=True).compute()
    df2.to_parquet(path, engine="pyarrow")

    ddf3 = dask_cudf.read_parquet(path, index=False)
    dd.assert_eq(df2, ddf3)


@pytest.mark.parametrize("write_meta", [True, False])
def test_roundtrip_from_dask_cudf(tmpdir, write_meta):
    tmpdir = str(tmpdir)
    gddf = ddf.to_backend("cudf")
    gddf.to_parquet(tmpdir, write_metadata_file=write_meta)

    gddf2 = dask_cudf.read_parquet(tmpdir, calculate_divisions=True)
    dd.assert_eq(gddf, gddf2)


def test_roundtrip_none_rangeindex(tmpdir):
    fn = str(tmpdir.join("test.parquet"))
    gdf = cudf.DataFrame(
        {"id": [0, 1, 2, 3], "val": [None, None, 0, 1]},
        index=pd.RangeIndex(start=5, stop=9),
    )
    dask_cudf.from_cudf(gdf, npartitions=2).to_parquet(fn)
    ddf2 = dask_cudf.read_parquet(fn)
    dd.assert_eq(gdf, ddf2, check_index=True)


def test_roundtrip_from_pandas(tmpdir):
    fn = str(tmpdir.join("test.parquet"))

    # First without specifying an index
    dfp = df.copy()
    dfp.to_parquet(fn, engine="pyarrow", index=False)
    dfp = dfp.reset_index(drop=True)
    ddf2 = dask_cudf.read_parquet(fn)
    dd.assert_eq(dfp, ddf2, check_index=True)

    # Now, specifying an index
    dfp = df.copy()
    dfp.to_parquet(fn, engine="pyarrow", index=True)
    ddf2 = dask_cudf.read_parquet(fn, index=["index"])
    dd.assert_eq(dfp, ddf2, check_index=True)


def test_strings(tmpdir):
    fn = str(tmpdir)
    dfp = pd.DataFrame(
        {"a": ["aa", "bbb", "cccc"], "b": ["hello", "dog", "man"]}
    )
    dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)
    ddf2.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn, index=["a"])
    dd.assert_eq(ddf2, read_df.compute().to_pandas())


def test_dask_timeseries_from_pandas(tmpdir):
    fn = str(tmpdir.join("test.parquet"))
    ddf2 = dask.datasets.timeseries(freq="D", seed=1)
    pdf = ddf2.compute()
    pdf.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    dd.assert_eq(ddf2, read_df.compute())


@pytest.mark.parametrize("index", [False, None])
@pytest.mark.parametrize("divisions", [False, True])
def test_dask_timeseries_from_dask(tmpdir, index, divisions):
    fn = str(tmpdir)
    ddf2 = dask.datasets.timeseries(freq="D", seed=1)
    ddf2.to_parquet(fn, engine="pyarrow", write_index=index)
    read_df = dask_cudf.read_parquet(
        fn, index=index, calculate_divisions=divisions
    )
    dd.assert_eq(
        ddf2, read_df, check_divisions=(divisions and index), check_index=index
    )


@pytest.mark.parametrize("index", [False, None])
@pytest.mark.parametrize("divisions", [False, True])
def test_dask_timeseries_from_daskcudf(tmpdir, index, divisions):
    fn = str(tmpdir)
    ddf2 = dask_cudf.from_cudf(
        cudf.datasets.timeseries(freq="D"), npartitions=4
    )
    # Use assign in lieu of `ddf2.name = ...`
    # See: https://github.com/dask/dask-expr/issues/1010
    ddf2 = ddf2.assign(name=ddf2.name.astype("object"))
    ddf2.to_parquet(fn, write_index=index)
    read_df = dask_cudf.read_parquet(
        fn, index=index, calculate_divisions=divisions
    )
    dd.assert_eq(
        ddf2, read_df, check_divisions=(divisions and index), check_index=index
    )


@pytest.mark.parametrize("index", [False, True])
def test_empty(tmpdir, index):
    fn = str(tmpdir)
    dfp = pd.DataFrame({"a": [11.0, 12.0, 12.0], "b": [4, 5, 6]})[:0]
    if index:
        dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)

    ddf2.to_parquet(fn, write_index=index, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    dd.assert_eq(ddf2, read_df.compute())


def test_filters(tmpdir):
    tmp_path = str(tmpdir)
    df = pd.DataFrame({"x": range(10), "y": list("aabbccddee")})
    ddf = dd.from_pandas(df, npartitions=5)
    assert ddf.npartitions == 5

    ddf.to_parquet(tmp_path, engine="pyarrow")

    a = dask_cudf.read_parquet(
        tmp_path, filters=[("x", ">", 4)], split_row_groups=True
    )
    assert a.npartitions == 3
    assert (a.x > 3).all().compute()

    b = dask_cudf.read_parquet(
        tmp_path, filters=[("y", "==", "c")], split_row_groups=True
    )
    assert b.npartitions == 1
    b = b.compute().to_pandas()
    assert (b.y == "c").all()

    c = dask_cudf.read_parquet(
        tmp_path,
        filters=[("y", "==", "c"), ("x", ">", 6)],
        split_row_groups=True,
    )
    assert c.npartitions <= 1
    assert not len(c)


@pytest.mark.parametrize("numeric", [True, False])
@pytest.mark.parametrize("null", [np.nan, None])
def test_isna_filters(tmpdir, null, numeric):
    tmp_path = str(tmpdir)
    df = pd.DataFrame(
        {
            "x": range(10),
            "y": list("aabbccddee"),
            "i": [0] * 4 + [np.nan] * 2 + [0] * 4,
            "j": [""] * 4 + [None] * 2 + [""] * 4,
        }
    )
    ddf = dd.from_pandas(df, npartitions=5)
    assert ddf.npartitions == 5
    ddf.to_parquet(tmp_path, engine="pyarrow")

    # Test "is"
    col = "i" if numeric else "j"
    filters = [(col, "is", null)]
    out = dask_cudf.read_parquet(
        tmp_path, filters=filters, split_row_groups=True
    )
    assert len(out) == 2
    assert list(out.x.compute().values) == [4, 5]

    # Test "is not"
    filters = [(col, "is not", null)]
    out = dask_cudf.read_parquet(
        tmp_path, filters=filters, split_row_groups=True
    )
    assert len(out) == 8
    assert list(out.x.compute().values) == [0, 1, 2, 3, 6, 7, 8, 9]


def test_filters_at_row_group_level(tmpdir):
    tmp_path = str(tmpdir)
    df = pd.DataFrame({"x": range(10), "y": list("aabbccddee")})
    ddf = dd.from_pandas(df, npartitions=5)
    assert ddf.npartitions == 5

    ddf.to_parquet(tmp_path, engine="pyarrow", row_group_size=10 / 5)

    a = dask_cudf.read_parquet(
        tmp_path, filters=[("x", "==", 1)], split_row_groups=True
    )
    assert a.npartitions == 1
    assert (a.shape[0] == 1).compute()

    # Overwrite=True can be removed for dask-expr>=0.4.1
    # See: https://github.com/dask-contrib/dask-expr/issues/800
    ddf.to_parquet(
        tmp_path, engine="pyarrow", row_group_size=1, overwrite=True
    )

    b = dask_cudf.read_parquet(
        tmp_path, filters=[("x", "==", 1)], split_row_groups=True
    )
    assert b.npartitions == 1
    assert (b.shape[0] == 1).compute()


@pytest.mark.parametrize("metadata", [True, False])
@pytest.mark.parametrize("daskcudf", [True, False])
@pytest.mark.parametrize(
    "parts", [["year", "month", "day"], ["year", "month"], ["year"]]
)
def test_roundtrip_from_dask_partitioned(tmpdir, parts, daskcudf, metadata):
    tmpdir = str(tmpdir)

    df = pd.DataFrame()
    df["year"] = [2018, 2019, 2019, 2019, 2020, 2021]
    df["month"] = [1, 2, 3, 3, 3, 2]
    df["day"] = [1, 1, 1, 2, 2, 1]
    df["data"] = [0, 0, 0, 0, 0, 0]
    df.index.name = "index"
    if daskcudf:
        ddf2 = dask_cudf.from_cudf(cudf.from_pandas(df), npartitions=2)
        ddf2.to_parquet(
            tmpdir, write_metadata_file=metadata, partition_on=parts
        )
    else:
        ddf2 = dd.from_pandas(df, npartitions=2)
        ddf2.to_parquet(
            tmpdir,
            engine="pyarrow",
            write_metadata_file=metadata,
            partition_on=parts,
        )
    df_read = dd.read_parquet(tmpdir, engine="pyarrow")
    gdf_read = dask_cudf.read_parquet(tmpdir)

    # TODO: Avoid column selection after `CudfEngine`
    # can be aligned with dask/dask#6534
    columns = list(df_read.columns)
    assert set(df_read.columns) == set(gdf_read.columns)
    dd.assert_eq(
        df_read.compute(scheduler=dask.get)[columns],
        gdf_read.compute(scheduler=dask.get)[columns],
    )

    assert gdf_read.index.name == "index"

    # Check that we don't have uuid4 file names
    for _, _, files in os.walk(tmpdir):
        for fn in files:
            if not fn.startswith("_"):
                assert "part" in fn

    # Check that we can aggregate files by a partition name
    df_read = dd.read_parquet(
        tmpdir, engine="pyarrow", aggregate_files="year", split_row_groups=2
    )
    gdf_read = dask_cudf.read_parquet(
        tmpdir, aggregate_files="year", split_row_groups=2
    )
    dd.assert_eq(df_read, gdf_read)


@pytest.mark.parametrize("row_groups", [1, 3, 10, 12])
@pytest.mark.parametrize("index", [False, True])
def test_split_row_groups(tmpdir, row_groups, index):
    nparts = 2
    df_size = 100
    row_group_size = 5
    file_row_groups = 10  # Known apriori
    npartitions_expected = math.ceil(file_row_groups / row_groups) * 2
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "a": rng.choice(["apple", "banana", "carrot"], size=df_size),
            "b": rng.random(size=df_size),
            "c": rng.integers(1, 5, size=df_size),
            "index": np.arange(0, df_size),
        }
    )
    if index:
        df = df.set_index("index")

    ddf1 = dd.from_pandas(df, npartitions=nparts)
    ddf1.to_parquet(
        str(tmpdir),
        engine="pyarrow",
        row_group_size=row_group_size,
        write_metadata_file=True,
    )

    ddf2 = dask_cudf.read_parquet(
        str(tmpdir),
        split_row_groups=row_groups,
    )

    dd.assert_eq(ddf1, ddf2, check_divisions=False)

    assert ddf2.npartitions == npartitions_expected


@need_create_meta
@pytest.mark.parametrize("partition_on", [None, "a"])
def test_create_metadata_file(tmpdir, partition_on):
    tmpdir = str(tmpdir)

    # Write ddf without a _metadata file
    df1 = cudf.DataFrame({"b": range(100), "a": ["A", "B", "C", "D"] * 25})
    df1.index.name = "myindex"
    ddf1 = dask_cudf.from_cudf(df1, npartitions=10)
    ddf1.to_parquet(
        tmpdir,
        write_metadata_file=False,
        partition_on=partition_on,
    )

    # Add global _metadata file
    if partition_on:
        fns = glob.glob(os.path.join(tmpdir, partition_on + "=*/*.parquet"))
    else:
        fns = glob.glob(os.path.join(tmpdir, "*.parquet"))

    with pytest.warns(
        match="dask_cudf.io.parquet.create_metadata_file is now deprecated"
    ):
        dask_cudf.io.parquet.create_metadata_file(
            fns,
            split_every=3,  # Force tree reduction
        )

    # Check that we can now read the ddf
    # with the _metadata file present
    ddf2 = dask_cudf.read_parquet(
        tmpdir,
        split_row_groups=False,
        index="myindex",
        calculate_divisions=True,
    )
    if partition_on:
        ddf1 = df1.sort_values("b")
        ddf2 = ddf2.compute().sort_values("b")
        ddf2.a = ddf2.a.astype("object")
    dd.assert_eq(ddf1, ddf2)


@need_create_meta
def test_create_metadata_file_inconsistent_schema(tmpdir):
    # NOTE: This test demonstrates that the CudfEngine
    # can be used to generate a global `_metadata` file
    # even if there are inconsistent schemas in the dataset.

    # Write file 0
    df0 = pd.DataFrame({"a": [None] * 10, "b": range(10)})
    p0 = os.path.join(tmpdir, "part.0.parquet")
    df0.to_parquet(p0, engine="pyarrow")

    # Write file 1
    b = list(range(10))
    b[1] = None
    df1 = pd.DataFrame({"a": range(10), "b": b})
    p1 = os.path.join(tmpdir, "part.1.parquet")
    df1.to_parquet(p1, engine="pyarrow")

    # New pyarrow-dataset base can handle an inconsistent
    # schema (even without a _metadata file), but computing
    # and dtype validation may fail
    ddf1 = dask_cudf.read_parquet(str(tmpdir), calculate_divisions=True)

    # Add global metadata file.
    # Dask-CuDF can do this without requiring schema
    # consistency.
    create_metadata_file([p0, p1])

    # Check that we can still read the ddf
    # with the _metadata file present
    ddf2 = dask_cudf.read_parquet(str(tmpdir), calculate_divisions=True)

    # Check that the result is the same with and
    # without the _metadata file.  Note that we must
    # call `compute` on `ddf1`, because the dtype of
    # the inconsistent column ("a") may be "object"
    # before computing, and "int" after
    dd.assert_eq(ddf1.compute(), ddf2)
    dd.assert_eq(ddf1.compute(), ddf2.compute())


@pytest.mark.parametrize("specify_schema", [True, False])
def test_read_inconsistent_schema(tmpdir, specify_schema):
    if specify_schema:
        # If we specify the expected schema,
        # we also need to specify the partitioning.
        kwargs = {
            "dataset": {
                "schema": pa.schema(
                    [
                        ("id", pa.int64()),
                        ("text", pa.string()),
                        ("meta1", pa.struct([("field1", pa.string())])),
                    ]
                ),
                "partitioning": None,
            },
        }
    else:
        kwargs = {}

    records = [
        {"id": 123, "text": "foo"},
        {
            "text": "bar",
            "meta1": [{"field1": "cat"}],
            "id": 456,
        },
    ]
    columns = ["text", "id"]
    pd.DataFrame(records[:1]).to_parquet(tmpdir / "part.0.parquet")
    pd.DataFrame(records[1:]).to_parquet(tmpdir / "part.1.parquet")
    # Check that cuDF and Dask cuDF match
    dd.assert_eq(
        cudf.read_parquet(
            tmpdir, columns=columns, allow_mismatched_pq_schemas=True
        ),
        dask_cudf.read_parquet(tmpdir, columns=columns, **kwargs),
        check_index=False,
    )
    # Check that "pandas" and "cudf" backends match
    dd.assert_eq(
        dd.read_parquet(tmpdir, columns=columns),
        dask_cudf.read_parquet(tmpdir, columns=columns, **kwargs),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["dog", "cat", "fish"],
        [[0], [1, 2], [3]],
        [None, [1, 2], [3]],
        [{"f1": 1}, {"f1": 0, "f2": "dog"}, {"f2": "cat"}],
        [None, {"f1": 0, "f2": "dog"}, {"f2": "cat"}],
    ],
)
def test_cudf_dtypes_from_pandas(tmpdir, data):
    # Simple test that we can read in list and struct types
    fn = str(tmpdir.join("test.parquet"))
    dfp = pd.DataFrame({"data": data})
    dfp.to_parquet(fn, engine="pyarrow", index=True)
    # Use `split_row_groups=True` to avoid "fast path" where
    # schema is not is passed through in older Dask versions
    ddf2 = dask_cudf.read_parquet(fn, split_row_groups=True)
    dd.assert_eq(cudf.from_pandas(dfp), ddf2)


def test_cudf_list_struct_write(tmpdir):
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [[[1, 2]], [[2, 3]], None],
            "c": [[[["a", "z"]]], [[["b", "d", "e"]]], None],
        }
    )
    df["d"] = df.to_struct()

    ddf = dask_cudf.from_cudf(df, 3)
    temp_file = str(tmpdir.join("list_struct.parquet"))

    ddf.to_parquet(temp_file)
    new_ddf = dask_cudf.read_parquet(temp_file)
    dd.assert_eq(df, new_ddf)


def test_null_partition(tmpdir):
    from pyarrow.dataset import HivePartitioning

    ids = pd.Series([0, 1, None], dtype="Int64")
    df = pd.DataFrame({"id": ids, "x": [1, 2, 3]})
    ddf = dd.from_pandas(df, npartitions=1).to_backend("cudf")
    ddf.to_parquet(str(tmpdir), partition_on="id")
    fns = glob.glob(os.path.join(tmpdir, "id" + "=*/*.parquet"))
    assert len(fns) == 3

    partitioning = HivePartitioning(pa.schema([("id", pa.int64())]))
    ddf_read = dask_cudf.read_parquet(
        str(tmpdir),
        dataset={"partitioning": partitioning},
    )
    dd.assert_eq(
        ddf[["x", "id"]],
        ddf_read[["x", "id"]],
        check_divisions=False,
        # Int64 is cast to float64 in dask
        check_dtype=False,
    )


def test_nullable_schema_mismatch(tmpdir):
    # See: https://github.com/NVIDIA/cudf/issues/12702
    path0 = str(tmpdir.join("test.0.parquet"))
    path1 = str(tmpdir.join("test.1.parquet"))
    cudf.DataFrame.from_dict({"a": [1, 2, 3]}).to_parquet(path0)
    cudf.DataFrame.from_dict({"a": [4, 5, None]}).to_parquet(path1)
    ddf = dask_cudf.read_parquet(
        [path0, path1], split_row_groups=2, aggregate_files=True
    )
    expect = pd.read_parquet([path0, path1])
    dd.assert_eq(ddf, expect, check_index=False)


def test_parquet_read_filter_and_project(tmpdir):
    # Filter on columns that are not included
    # in the current column projection

    # Write parquet data
    path = str(tmpdir.join("test.parquet"))
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 10,
            "b": [0, 1, 2, 3, 4] * 10,
            "c": range(50),
            "d": [6, 7] * 25,
            "e": [8, 9] * 25,
        }
    )
    df.to_parquet(path)

    # Read back with filter and projection
    columns = ["b"]
    filters = [[("a", "==", 5), ("c", ">", 20)]]
    got = dask_cudf.read_parquet(path, columns=columns, filters=filters)

    # Check result
    expected = df[(df.a == 5) & (df.c > 20)][columns].reset_index(drop=True)
    dd.assert_eq(got, expected)


def test_timezone_column(tmpdir):
    path = str(tmpdir.join("test.parquet"))
    pdf = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["1996-01-02", "1996-12-01"],
                utc=True,
            ),
            "x": [1, 2],
        }
    )
    pdf.to_parquet(path)

    # Check that `cudf` and `dask_cudf` results match
    got = dask_cudf.read_parquet(path)
    expect = cudf.read_parquet(path)
    dd.assert_eq(got, expect)


@pytest.mark.parametrize("min_part_size", ["1B", "1GB"])
def test_read_parquet_arrow_filesystem(tmpdir, min_part_size):
    tmp_path = str(tmpdir)
    with dask.config.set(
        {
            "dataframe.backend": "cudf",
            "dataframe.parquet.minimum-partition-size": min_part_size,
        }
    ):
        dd.from_dict(
            {"x": range(1000), "y": ["a", "b", "c", "d"] * 250},
            npartitions=10,
        ).to_parquet(tmp_path, write_index=False)
        df = cudf.read_parquet(tmp_path)
        ddf = dask_cudf.read_parquet(tmp_path, filesystem="arrow")
        dd.assert_eq(df, ddf, check_index=False)
        assert isinstance(ddf._meta, cudf.DataFrame)
        assert isinstance(ddf.compute(), cudf.DataFrame)


@pytest.mark.parametrize("write_metadata_file", [True, False])
def test_to_parquet_append(tmpdir, write_metadata_file):
    df = cudf.DataFrame({"a": [1, 2, 3]})
    ddf = dask_cudf.from_cudf(df, npartitions=1)
    ddf.to_parquet(
        tmpdir,
        append=True,
        write_metadata_file=write_metadata_file,
        write_index=False,
    )
    ddf.to_parquet(
        tmpdir,
        append=True,
        write_metadata_file=write_metadata_file,
        write_index=False,
    )
    ddf2 = dask_cudf.read_parquet(tmpdir, blocksize=None)
    dd.assert_eq(cudf.concat([df, df]), ddf2)


def test_deprecated_api_paths(tmpdir):
    df = dask_cudf.DataFrame.from_dict({"a": range(100)}, npartitions=1)
    # io.to_parquet function is deprecated
    with pytest.warns(match="dask_cudf.io.to_parquet is now deprecated"):
        dask_cudf.io.to_parquet(df, tmpdir)

    df2 = dask_cudf.io.read_parquet(tmpdir)
    dd.assert_eq(df, df2, check_divisions=False)

    df2 = dask_cudf.io.parquet.read_parquet(tmpdir)
    dd.assert_eq(df, df2, check_divisions=False)
