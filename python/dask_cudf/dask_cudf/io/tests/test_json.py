# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pandas as pd
import pytest

import dask
import dask.dataframe as dd
from dask.utils import tmpfile

import dask_cudf


def test_read_json_backend_dispatch(tmp_path):
    # Test ddf.read_json cudf-backend dispatch
    df1 = dask.datasets.timeseries(
        dtypes={"x": int, "y": int},
        freq="120s",
        seed=1,
    ).reset_index(drop=True)
    json_path = str(tmp_path / "data-*.json")
    df1.to_json(json_path)
    with dask.config.set({"dataframe.backend": "cudf"}):
        df2 = dd.read_json(json_path)
    assert isinstance(df2, dask_cudf.DataFrame)
    dd.assert_eq(df1, df2)


def test_read_json(tmp_path):
    df1 = dask.datasets.timeseries(
        dtypes={"x": int, "y": int},
        freq="120s",
        seed=1,
    ).reset_index(drop=True)
    json_path = str(tmp_path / "data-*.json")
    df1.to_json(json_path)
    df2 = dask_cudf.read_json(json_path)
    dd.assert_eq(df1, df2)

    # file path test
    stmp_path = str(tmp_path / "data-*.json")
    df3 = dask_cudf.read_json(f"file://{stmp_path}")
    dd.assert_eq(df1, df3)

    # file list test
    list_paths = [
        os.path.join(tmp_path, fname) for fname in sorted(os.listdir(tmp_path))
    ]
    df4 = dask_cudf.read_json(list_paths)
    dd.assert_eq(df1, df4)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("orient", ["split", "index", "columns", "values"])
def test_read_json_basic(orient):
    df = pd.DataFrame({"x": ["a", "b", "c", "d"], "y": [1, 2, 3, 4]})
    with tmpfile("json") as f:
        df.to_json(f, orient=orient, lines=False)
        actual = dask_cudf.read_json(f, orient=orient, lines=False)
        actual_pd = pd.read_json(f, orient=orient, lines=False)
        dd.assert_eq(actual, actual_pd)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("lines", [True, False])
def test_read_json_lines(lines):
    df = pd.DataFrame({"x": ["a", "b", "c", "d"], "y": [1, 2, 3, 4]})
    with tmpfile("json") as f:
        df.to_json(f, orient="records", lines=lines)
        actual = dask_cudf.read_json(f, orient="records", lines=lines)
        actual_pd = pd.read_json(f, orient="records", lines=lines)
        dd.assert_eq(actual, actual_pd)


def test_read_json_nested(tmp_path):
    # Check that `engine="cudf"` can
    # be used to support nested data
    df = pd.DataFrame(
        {
            "a": [{"y": 2}, {"y": 4}, {"y": 6}, {"y": 8}],
            "b": [[1, 2, 3], [4, 5], [6], [7]],
            "c": [1, 3, 5, 7],
        }
    )
    kwargs = dict(orient="records", lines=True)
    f = tmp_path / "data.json"
    with dask.config.set({"dataframe.convert-string": False}):
        df.to_json(f, **kwargs)
        # Ensure engine='cudf' is tested.
        actual = dask_cudf.read_json(f, engine="cudf", **kwargs)
        actual_pd = pd.read_json(f, **kwargs)
        dd.assert_eq(actual, actual_pd)
        # Ensure not passing engine='cudf'(default i.e., auto) is tested.
        actual = dask_cudf.read_json(f, **kwargs)
        dd.assert_eq(actual, actual_pd)
        # Ensure not passing kwargs also reads the file.
        actual = dask_cudf.read_json(f)
        dd.assert_eq(actual, actual_pd)


def test_read_json_aggregate_files(tmp_path):
    df1 = dask.datasets.timeseries(
        dtypes={"x": int, "y": int},
        freq="120s",
        seed=1,
    ).reset_index(drop=True)
    json_path = str(tmp_path / "data-*.json")
    df1.to_json(json_path)

    df2 = dask_cudf.read_json(json_path, aggregate_files=2)
    assert df2.npartitions == math.ceil(df1.npartitions / 2)
    dd.assert_eq(df1, df2, check_index=False)

    df2 = dask_cudf.read_json(
        json_path, aggregate_files=True, blocksize="1GiB"
    )
    assert df2.npartitions == 1
    dd.assert_eq(df1, df2, check_index=False)

    for include_path_column, name in [(True, "path"), ("file", "file")]:
        df2 = dask_cudf.read_json(
            json_path,
            aggregate_files=2,
            include_path_column=include_path_column,
        )
        assert name in df2.columns
        assert len(df2[name].compute().unique()) == df1.npartitions
        dd.assert_eq(df1, df2.drop(columns=[name]), check_index=False)


def test_deprecated_api_paths(tmp_path):
    path = str(tmp_path / "data-*.json")
    df = dd.from_dict({"a": range(100)}, npartitions=1)
    df.to_json(path)

    # Encourage top-level read_json import only
    with pytest.warns(match="dask_cudf.io.read_json is now deprecated"):
        df2 = dask_cudf.io.read_json(path)
    dd.assert_eq(df, df2, check_divisions=False)
