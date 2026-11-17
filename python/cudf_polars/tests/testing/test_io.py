# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.io import make_partitioned_source


@pytest.mark.parametrize("fmt", ["csv", "ndjson", "parquet"])
@pytest.mark.parametrize("use_str_path", [True, False])
def test_make_source_single_file(tmp_path, fmt, use_str_path):
    df = pl.DataFrame({"a": [1, 2], "b": ["foo", "bar"]})
    path = tmp_path / f"df.{fmt}"
    make_partitioned_source(df, str(path) if use_str_path else path, fmt)

    assert path.exists()
    assert path.is_file()


@pytest.mark.parametrize("fmt", ["csv", "ndjson", "parquet"])
@pytest.mark.parametrize("n_files", [2, 5])
def test_make_source_multiple_files(tmp_path, fmt, n_files):
    df = pl.DataFrame({"a": list(range(100)), "b": ["x"] * 100})
    make_partitioned_source(df, tmp_path, fmt, n_files=n_files)

    for i, file in enumerate(sorted(tmp_path.iterdir())):
        assert file.exists()
        assert file.is_file()
        assert file.name == f"part.{i}.{fmt}"


def test_make_source_invalid_format(tmp_path):
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Unsupported format: foo"):
        make_partitioned_source(df, tmp_path / "bad.foo", "foo")


@pytest.mark.parametrize("fmt", ["csv", "ndjson", "parquet"])
def test_make_source_single_file_in_dir(tmp_path, fmt):
    # When n_files=1 and path is a directory, the file should be created inside it
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    make_partitioned_source(df, tmp_path, fmt)
    expected = tmp_path / f"part.0.{fmt}"
    assert expected.exists()
    assert expected.is_file()


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_make_lazy_frame_from_file(tmp_path, fmt):
    from cudf_polars.testing.io import make_lazy_frame

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    lf = make_lazy_frame(df, fmt, path=tmp_path / f"test.{fmt}")
    result = lf.collect()
    assert result.shape == (3, 2)


def test_make_lazy_frame_from_frame():
    from cudf_polars.testing.io import make_lazy_frame

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    lf = make_lazy_frame(df, "frame")
    assert lf.collect().equals(df)


def test_make_lazy_frame_from_frame_with_n_rows():
    from cudf_polars.testing.io import make_lazy_frame

    df = pl.DataFrame({"a": list(range(10))})
    lf = make_lazy_frame(df, "frame", n_rows=5)
    assert lf.collect().shape == (5, 1)


def test_make_lazy_frame_from_file_with_n_rows(tmp_path):
    from cudf_polars.testing.io import make_lazy_frame

    df = pl.DataFrame({"a": list(range(10))})
    lf = make_lazy_frame(df, "parquet", path=tmp_path / "test.parquet", n_rows=3)
    assert lf.collect().shape == (3, 1)
