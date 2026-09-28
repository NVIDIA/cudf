# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from contextlib import nullcontext
from datetime import datetime, timezone

import pytest

import dask
from dask import dataframe as dd

import cudf

import dask_cudf

cur_dir = os.path.dirname(__file__)
sample_orc = os.path.join(cur_dir, "data/orc/sample.orc")


def test_read_orc_backend_dispatch():
    # Test ddf.read_orc cudf-backend dispatch
    df1 = cudf.read_orc(sample_orc)
    with dask.config.set({"dataframe.backend": "cudf"}):
        df2 = dd.read_orc(sample_orc)
    assert isinstance(df2, dask_cudf.DataFrame)
    dd.assert_eq(df1, df2, check_index=False)


def test_read_orc_defaults():
    df1 = cudf.read_orc(sample_orc)
    df2 = dask_cudf.read_orc(sample_orc)
    dd.assert_eq(df1, df2, check_index=False)


def test_filepath_read_orc_defaults():
    path = "file://%s" % sample_orc
    df1 = cudf.read_orc(path)
    df2 = dask_cudf.read_orc(path)
    dd.assert_eq(df1, df2, check_index=False)


def test_filelist_read_orc_defaults():
    path = [sample_orc]
    df1 = cudf.read_orc(path[0])
    df2 = dask_cudf.read_orc(path)
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("engine", ["cudf", "pyarrow"])
@pytest.mark.parametrize("columns", [["time", "date"], ["time"]])
def test_read_orc_cols(engine, columns):
    if engine == "pyarrow":
        ctx = pytest.warns(
            UserWarning, match="Using CPU via PyArrow to read ORC dataset."
        )
    else:
        ctx = nullcontext()

    with ctx:
        df1 = cudf.read_orc(sample_orc, engine=engine, columns=columns)

    with ctx:
        df2 = dask_cudf.read_orc(sample_orc, engine=engine, columns=columns)

    with ctx:
        dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("engine", ["cudf", "pyarrow"])
@pytest.mark.parametrize(
    "predicate,expected_len",
    [
        (None, 70_000),
        (
            [("date", "==", datetime(1900, 12, 25, tzinfo=timezone.utc))],
            15_000,
        ),
        (
            [("date", "<=", datetime(1928, 12, 25, tzinfo=timezone.utc))],
            30_000,
        ),
        (
            [
                [("date", ">", datetime(1950, 12, 25, tzinfo=timezone.utc))],
                [("date", "<=", datetime(1928, 12, 25, tzinfo=timezone.utc))],
            ],
            55_000,
        ),
    ],
)
def test_read_orc_filtered(tmpdir, engine, predicate, expected_len):
    if engine == "pyarrow":
        ctx = pytest.warns(
            UserWarning, match="Using CPU via PyArrow to read ORC dataset."
        )
    else:
        ctx = nullcontext()

    with ctx:
        df = dask_cudf.read_orc(sample_orc, engine=engine, filters=predicate)

    with ctx:
        dd.assert_eq(len(df), expected_len)


def test_read_orc_first_file_empty(tmpdir):
    # Write a 3-file dataset where the first file is empty
    # See: https://github.com/NVIDIA/cudf/issues/8011
    path = str(tmpdir)
    os.makedirs(path, exist_ok=True)
    df1 = cudf.DataFrame({"id": [1, 2], "float": [1.0, 2.0]})
    df1.iloc[:0].to_orc(os.path.join(path, "data.0"))
    df1.iloc[:1].to_orc(os.path.join(path, "data.1"))
    df1.iloc[1:].to_orc(os.path.join(path, "data.2"))

    # Read back the files with dask_cudf,
    # and check the result.
    df2 = dask_cudf.read_orc(os.path.join(path, "*"))
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("compute", [True, False])
@pytest.mark.parametrize("compression", [None, "snappy"])
@pytest.mark.parametrize(
    "dtypes",
    [
        {"index": int, "c": int, "a": str},
        {"index": int, "c": int, "a": str, "b": float},
        {"index": int, "c": str, "a": object},
    ],
)
def test_to_orc(tmpdir, dtypes, compression, compute):
    # Create cudf and dask_cudf dataframes
    df = cudf.datasets.randomdata(nrows=10, dtypes=dtypes, seed=1)
    df = df.set_index("index").sort_index()
    ddf = dask_cudf.from_cudf(df, npartitions=3)

    # Write cudf dataframe as single file
    # (preserve index by setting to column)
    fname = tmpdir.join("test.orc")
    df.reset_index().to_orc(fname, compression=compression)

    # Write dask_cudf dataframe as multiple files
    # (preserve index by `write_index=True`)
    to = ddf.to_orc(
        str(tmpdir), write_index=True, compression=compression, compute=compute
    )

    if not compute:
        to.compute()

    # Read back cudf dataframe
    df_read = cudf.read_orc(fname).set_index("index")

    # Read back dask_cudf dataframe
    paths = glob.glob(str(tmpdir) + "/part.*.orc")
    ddf_read = dask_cudf.read_orc(paths).set_index("index")

    # Make sure the dask_cudf dataframe matches
    # the cudf dataframes (df and df_read)
    dd.assert_eq(df, ddf_read)
    dd.assert_eq(df_read, ddf_read)


def test_deprecated_api_paths(tmpdir):
    df = dask_cudf.DataFrame.from_dict({"a": range(100)}, npartitions=1)
    path = tmpdir.join("test.orc")
    # Top-level to_orc function is deprecated
    with pytest.warns(match="dask_cudf.to_orc is now deprecated"):
        dask_cudf.to_orc(df, path, write_index=False)

    # Encourage top-level read_orc import only
    paths = glob.glob(str(path) + "/*.orc")
    with pytest.warns(match="dask_cudf.io.read_orc is now deprecated"):
        df2 = dask_cudf.io.read_orc(paths)
    dd.assert_eq(df, df2, check_divisions=False)
