# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.utils import cucascade


class FakeScanCache:
    """Stands in for `cudf_polars_cucascade.ScanCache`."""

    def __init__(self):
        self.opened = []
        self.layouts = {}

    def open(self, path):
        self.opened.append(path)
        return types.SimpleNamespace(path=path, nbytes=lambda: 10)

    def register_layout(self, path, directory, columns, col_index, offsets, sizes):
        self.layouts[path] = (directory, columns, col_index, offsets, sizes)


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.setattr(cucascade, "_cache", None)


@pytest.fixture
def glue(monkeypatch):
    fake = FakeScanCache()
    monkeypatch.setattr(cucascade, "_cache", fake)
    monkeypatch.setattr(cucascade, "_registered", set())
    return fake


@pytest.fixture
def parquet_info(tmp_path):
    path = tmp_path / "t.parquet"
    pl.DataFrame({"a": range(1000), "b": [str(i) for i in range(1000)]}).write_parquet(
        path, row_group_size=250
    )
    source = plc.io.types.SourceInfo([str(path)])
    (file_metadata,) = plc.io.parquet_metadata.read_parquet_footers(source)
    return types.SimpleNamespace(
        path="s3://bucket/table/t.parquet",
        size=path.stat().st_size,
        file_metadata=file_metadata,
    )


def test_disabled_is_a_no_op(disabled):
    assert cucascade.wrap_sources(["s3://bucket/a.parquet"]) is None
    cucascade.register_layouts([])
    cucascade.configure_cucascade(0)
    assert cucascade._cache is None


def test_missing_extension_is_reported_with_how_to_build_it(disabled, monkeypatch):
    monkeypatch.delitem(sys.modules, "cudf_polars_cucascade", raising=False)
    monkeypatch.setattr(sys, "path", [])
    with pytest.raises(ImportError, match="cudf_polars_cucascade") as excinfo:
        cucascade.configure_cucascade(1 << 30)
    assert "pip install" in str(excinfo.value)
    assert "python/cudf_polars/cucascade" in str(excinfo.value)


def test_extension_built_against_another_libcudf_is_refused(disabled, monkeypatch):
    running = cucascade._installed_cudf_version()
    if running is None:
        pytest.skip("the installed libcudf version is not known")
    stale = types.ModuleType("cudf_polars_cucascade")
    stale.built_against_cudf_version = (running[0], running[1] + 1, 0)
    monkeypatch.setitem(sys.modules, "cudf_polars_cucascade", stale)
    with pytest.raises(ImportError, match="rebuild"):
        cucascade.configure_cucascade(1 << 30)
    assert cucascade._cache is None


def test_extension_built_against_this_libcudf_is_accepted(monkeypatch):
    running = cucascade._installed_cudf_version()
    if running is None:
        pytest.skip("the installed libcudf version is not known")
    current = types.ModuleType("cudf_polars_cucascade")
    current.built_against_cudf_version = (running[0], running[1], 0)
    monkeypatch.setitem(sys.modules, "cudf_polars_cucascade", current)
    assert cucascade._load_extension() is current


def test_wrap_sources_opens_every_remote_path(glue):
    paths = ["s3://bucket/a.parquet", "s3://bucket/b.parquet"]
    sources = cucascade.wrap_sources(paths)
    assert sources is not None
    assert [s.path for s in sources] == paths
    assert sorted(glue.opened) == paths


@pytest.mark.parametrize(
    "paths", [[], ["/local/a.parquet"], ["s3://bucket/a.parquet", "/local/b.parquet"]]
)
def test_wrap_sources_leaves_local_reads_alone(glue, paths):
    assert cucascade.wrap_sources(paths) is None
    assert glue.opened == []


def test_layout_covers_every_column_chunk(parquet_info):
    columns, col_index, offsets, sizes = cucascade._layout(parquet_info)
    assert columns == ["a", "b"]
    assert col_index == [0, 1] * 4  # four row groups, row-group-major
    assert all(size > 0 for size in sizes)
    assert max(o + s for o, s in zip(offsets, sizes, strict=True)) <= parquet_info.size
    assert offsets == sorted(offsets)


def test_register_layouts_registers_each_file_once(glue, parquet_info):
    cucascade.register_layouts([parquet_info, parquet_info])
    assert list(glue.layouts) == [parquet_info.path]
    directory, columns, *_ = glue.layouts[parquet_info.path]
    assert directory == "s3://bucket/table"
    assert columns == ["a", "b"]


def test_register_layouts_skips_local_files(glue, parquet_info):
    parquet_info.path = "/local/t.parquet"
    cucascade.register_layouts([parquet_info])
    assert glue.layouts == {}
