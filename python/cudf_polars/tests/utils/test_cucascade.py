# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.utils import cucascade


class FakeGlue:
    """Stands in for the compiled `_cucascade_io` module."""

    def __init__(self):
        self.opened = []
        self.layouts = {}

    def CucascadeDatasource(self, path, size):
        self.opened.append((path, size))
        return types.SimpleNamespace(path=path, nbytes=lambda: size)

    def register_layout(self, path, directory, columns, col_index, offsets, sizes):
        self.layouts[path] = (directory, columns, col_index, offsets, sizes)


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.setattr(cucascade, "_module", None)


@pytest.fixture
def glue(monkeypatch):
    fake = FakeGlue()
    monkeypatch.setattr(cucascade, "_module", fake)
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
    assert cucascade._module is None


def test_missing_glue_is_reported(disabled, monkeypatch):
    monkeypatch.delitem(sys.modules, "_cucascade_io", raising=False)
    monkeypatch.setattr(sys, "path", [])
    with pytest.raises(RuntimeError, match="_cucascade_io"):
        cucascade.configure_cucascade(1 << 30)


def test_wrap_sources_opens_every_remote_path(glue):
    paths = ["s3://bucket/a.parquet", "s3://bucket/b.parquet"]
    sources = cucascade.wrap_sources(paths, [10, None])
    assert sources is not None
    assert [s.path for s in sources] == paths
    assert sorted(glue.opened) == [(paths[0], 10), (paths[1], 0)]


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
