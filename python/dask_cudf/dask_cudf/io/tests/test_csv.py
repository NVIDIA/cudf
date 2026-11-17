# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd

import cudf

import dask_cudf


@pytest.fixture
def csv_begin_bad_lines(tmp_path):
    lines = """x
    x
    x
    A, B, C, D
    1, 2, 3, 4
    2, 3, 5, 1
    4, 5, 2, 5"""

    file = tmp_path / "test_read_csv_begin.csv"

    with open(file, "w") as fp:
        fp.write(lines)

    return file


@pytest.fixture
def csv_end_bad_lines(tmp_path):
    lines = """A, B, C, D
    1, 2, 3, 4
    2, 3, 5, 1
    4, 5, 2, 5
    x
    x
    x"""

    file = tmp_path / "test_read_csv_end.csv"

    with open(file, "w") as fp:
        fp.write(lines)

    return file


def test_csv_roundtrip_backend_dispatch(tmp_path):
    # Test ddf.read_csv cudf-backend dispatch
    df = cudf.DataFrame({"x": [1, 2, 3, 4], "id": ["a", "b", "c", "d"]})
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    csv_path = str(tmp_path / "data-*.csv")
    ddf.to_csv(csv_path, index=False)
    with dask.config.set({"dataframe.backend": "cudf"}):
        ddf2 = dd.read_csv(csv_path)
    assert isinstance(ddf2, dask_cudf.DataFrame)
    dd.assert_eq(ddf, ddf2, check_divisions=False, check_index=False)


def test_csv_roundtrip(tmp_path):
    df = cudf.DataFrame({"x": [1, 2, 3, 4], "id": ["a", "b", "c", "d"]})
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    csv_path = str(tmp_path / "data-*.csv")
    ddf.to_csv(csv_path, index=False)

    ddf2 = dask_cudf.read_csv(csv_path)
    dd.assert_eq(ddf, ddf2, check_divisions=False, check_index=False)


def test_csv_roundtrip_filepath(tmp_path):
    df = cudf.DataFrame({"x": [1, 2, 3, 4], "id": ["a", "b", "c", "d"]})
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    stmp_path = str(tmp_path / "data-*.csv")

    ddf.to_csv(f"file://{stmp_path}", index=False)

    ddf2 = dask_cudf.read_csv(f"file://{stmp_path}")
    dd.assert_eq(ddf, ddf2, check_divisions=False, check_index=False)


def test_read_csv(tmp_path):
    df = dask.datasets.timeseries(
        dtypes={"x": int, "y": int},
        freq="120s",
        seed=1,
    ).reset_index(drop=True)

    csv_path = str(tmp_path / "data-*.csv")
    df.to_csv(csv_path, index=False)

    df2 = dask_cudf.read_csv(csv_path)
    dd.assert_eq(df, df2)

    # file path test
    stmp_path = str(csv_path)
    df3 = dask_cudf.read_csv(f"file://{stmp_path}")
    dd.assert_eq(df2, df3)

    # file list test
    list_paths = [
        os.path.join(tmp_path, fname) for fname in sorted(os.listdir(tmp_path))
    ]
    df4 = dask_cudf.read_csv(list_paths)
    dd.assert_eq(df, df4)


def test_raises_FileNotFoundError():
    with pytest.raises(FileNotFoundError):
        dask_cudf.read_csv("foo.csv")


def test_read_csv_w_bytes(tmp_path):
    df = dask.datasets.timeseries(
        dtypes={"x": int, "y": int},
        freq="120s",
        seed=1,
    ).reset_index(drop=True)
    df = pd.DataFrame(dict(x=np.arange(20), y=np.arange(20)))
    df.to_csv(tmp_path / "data-*.csv", index=False)

    df2 = dask_cudf.read_csv(tmp_path / "*.csv", blocksize="50 B")
    assert df2.npartitions == 3
    dd.assert_eq(df2, df, check_index=False)


def test_read_csv_compression(tmp_path):
    df = pd.DataFrame(dict(x=np.arange(20), y=np.arange(20)))
    df.to_csv(tmp_path / "data.csv.gz", index=False)

    with pytest.warns(UserWarning) as w:
        df2 = dask_cudf.read_csv(tmp_path / "*.csv.gz", blocksize="50 B")

    assert len(w) == 1
    msg = str(w[0].message)
    assert "gzip" in msg

    assert df2.npartitions == 1
    dd.assert_eq(df2, df, check_index=False)

    with warnings.catch_warnings(record=True) as record:
        df2 = dask_cudf.read_csv(tmp_path / "*.csv.gz", blocksize=None)

        assert not record


def test_read_csv_compression_file_list(tmp_path):
    # Repro from Issue#3412
    lines = """col1,col2
    0,1
    2,3"""

    files = [tmp_path / "test1.csv", tmp_path / "test2.csv"]

    for fn in files:
        with gzip.open(fn, "wb") as fp:
            fp.write(lines.encode("utf-8"))

    ddf_cpu = dd.read_csv(files, compression="gzip", blocksize=None).compute()
    ddf_gpu = dask_cudf.read_csv(
        files, compression="gzip", blocksize=None
    ).compute()

    dd.assert_eq(ddf_cpu, ddf_gpu)


@pytest.mark.parametrize("size", [0, 3, 20])
@pytest.mark.parametrize("compression", ["gzip", None])
def test_read_csv_blocksize_none(tmp_path, compression, size):
    df = pd.DataFrame(dict(x=np.arange(size), y=np.arange(size)))

    path = (
        tmp_path / "data.csv.gz"
        if compression == "gzip"
        else tmp_path / "data.csv"
    )

    # Types need to be specified for empty csv files
    if size == 0:
        typ = {"x": df.x.dtype, "y": df.y.dtype}
    else:
        typ = None

    df.to_csv(path, index=False, compression=compression)
    df2 = dask_cudf.read_csv(path, blocksize=None, dtype=typ)
    dd.assert_eq(df, df2)


@pytest.mark.parametrize("dtype", [{"b": str, "c": int}, None])
def test_csv_reader_usecols(tmp_path, dtype):
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4] * 100,
            "b": ["a", "b", "c", "d"] * 100,
            "c": [10, 11, 12, 13] * 100,
        }
    )
    csv_path = str(tmp_path / "usecols_data.csv")
    df.to_csv(csv_path, index=False)
    ddf = dask_cudf.from_cudf(df[["b", "c"]], npartitions=5)
    ddf2 = dask_cudf.read_csv(csv_path, usecols=["b", "c"], dtype=dtype)

    dd.assert_eq(ddf, ddf2, check_divisions=False, check_index=False)


def test_read_csv_skiprows(csv_begin_bad_lines):
    # Repro from Issue#13552
    ddf_cpu = dd.read_csv(csv_begin_bad_lines, skiprows=3).compute()
    ddf_gpu = dask_cudf.read_csv(csv_begin_bad_lines, skiprows=3).compute()

    dd.assert_eq(ddf_cpu, ddf_gpu)


def test_read_csv_skiprows_error(csv_begin_bad_lines):
    # Repro from Issue#13552
    with pytest.raises(ValueError):
        dask_cudf.read_csv(
            csv_begin_bad_lines, skiprows=3, blocksize="100 MiB"
        ).compute()


def test_read_csv_skipfooter(csv_end_bad_lines):
    # Repro from Issue#13552
    # Use pd.read_csv as the reference: dd.read_csv has a meta-inference
    # issue where it reads the first few rows without skipfooter (exposing
    # the bad footer rows), causing column-A dtype to be inferred as string
    # rather than int. pd.read_csv and dask_cudf.read_csv both produce the
    # correct result.
    with dask.config.set({"dataframe.convert-string": False}):
        ddf_cpu = pd.read_csv(csv_end_bad_lines, skipfooter=3, engine="python")
        ddf_gpu = dask_cudf.read_csv(csv_end_bad_lines, skipfooter=3).compute()

        dd.assert_eq(ddf_cpu, ddf_gpu, check_dtype=False)


def test_read_csv_skipfooter_arrow_string_fail(request, csv_end_bad_lines):
    request.applymarker(
        pytest.mark.xfail(
            reason="https://github.com/NVIDIA/cudf/issues/14915",
        )
    )
    ddf_cpu = dd.read_csv(
        csv_end_bad_lines, skipfooter=3, engine="python"
    ).compute()
    ddf_gpu = dask_cudf.read_csv(csv_end_bad_lines, skipfooter=3).compute()

    dd.assert_eq(ddf_cpu, ddf_gpu, check_dtype=False)


def test_read_csv_skipfooter_error(csv_end_bad_lines):
    with pytest.raises(ValueError):
        dask_cudf.read_csv(
            csv_end_bad_lines, skipfooter=3, blocksize="100 MiB"
        ).compute()


def test_read_csv_nrows(csv_end_bad_lines):
    ddf_cpu = pd.read_csv(csv_end_bad_lines, nrows=2)
    ddf_gpu = dask_cudf.read_csv(csv_end_bad_lines, nrows=2).compute()

    dd.assert_eq(ddf_cpu, ddf_gpu)


def test_read_csv_nrows_error(csv_end_bad_lines):
    with pytest.raises(ValueError):
        dask_cudf.read_csv(
            csv_end_bad_lines, nrows=2, blocksize="100 MiB"
        ).compute()


def test_deprecated_api_paths(tmp_path):
    csv_path = str(tmp_path / "data-*.csv")
    df = dask_cudf.DataFrame.from_dict({"a": range(100)}, npartitions=1)
    df.to_csv(csv_path, index=False)

    # Encourage top-level read_csv import only
    with pytest.warns(match="dask_cudf.io.read_csv is now deprecated"):
        df2 = dask_cudf.io.read_csv(csv_path)
    dd.assert_eq(df, df2, check_divisions=False)
