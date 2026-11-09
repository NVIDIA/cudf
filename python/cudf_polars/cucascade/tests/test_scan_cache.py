# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import pytest

import cudf_polars_cucascade
from cudf_polars_cucascade import ColumnStats, ScanCache, ScanCacheOptions

LAYOUT = {
    "table": "s3://bucket/t",
    "columns": ["a", "b"],
    "column_of_chunk": [0, 1],
    "offsets": [0, 4000],
    "sizes": [4000, 4000],
}


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        ScanCacheOptions(
            256 << 20, "http://127.0.0.1:9", "us-east-1", "k", "s"
        )


def test_options_need_the_credentials():
    with pytest.raises(TypeError):
        ScanCacheOptions(cache_bytes=1 << 20)


def test_a_scan_cache_needs_a_size(make_options):
    with pytest.raises(ValueError, match="cache_bytes"):
        ScanCache(make_options(cache_bytes=0))


@pytest.mark.parametrize("numa_node", [None, 0, -1])
def test_the_cache_can_be_placed_on_a_numa_node(make_options, numa_node):
    assert (
        ScanCache(make_options(numa_node=numa_node))
        .summary()
        .startswith("fs_cache:")
    )


def test_a_scan_cache_needs_options():
    with pytest.raises(TypeError):
        ScanCache(None)


def test_a_registered_layout_shows_up_in_the_statistics(scan_cache):
    scan_cache.register_layout("s3://bucket/t/f1.parquet", **LAYOUT)
    stats = {s.name: s for s in scan_cache.column_stats()}
    assert all(isinstance(s, ColumnStats) for s in stats.values())
    assert stats["s3://bucket/t/a"].registered == 4000
    assert stats["s3://bucket/t/b"].registered == 4000
    assert stats["s3://bucket/t/a"].requested == 0


def test_registering_a_file_twice_has_no_effect(scan_cache):
    layout = {**LAYOUT, "table": "s3://bucket/u"}
    scan_cache.register_layout("s3://bucket/u/f1.parquet", **layout)
    scan_cache.register_layout("s3://bucket/u/f1.parquet", **layout)
    scan_cache.register_layout("s3://bucket/u/f2.parquet", **layout)
    stats = {s.name: s for s in scan_cache.column_stats()}
    # Two files, 4000 bytes of column a in each.
    assert stats["s3://bucket/u/a"].registered == 8000


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"sizes": [10, 10, 10]}, "same length"),
        ({"column_of_chunk": [0, 7]}, "out of range"),
    ],
)
def test_a_malformed_layout_is_rejected(scan_cache, overrides, match):
    with pytest.raises(ValueError, match=match):
        scan_cache.register_layout(
            "s3://bucket/t/bad.parquet", **{**LAYOUT, **overrides}
        )


def test_the_summary_reports_the_prefetching_cache(scan_cache):
    assert scan_cache.summary().startswith("fs_cache:")


def test_opening_an_unsupported_path_raises_a_value_error(scan_cache):
    with pytest.raises(ValueError):
        scan_cache.open("ftp://bucket/a.parquet")


def test_the_libcudf_it_was_built_against_is_recorded():
    built = cudf_polars_cucascade.built_against_cudf_version
    assert len(built) == 3
    assert all(isinstance(part, int) for part in built)
    major, minor = (int(part) for part in version("pylibcudf").split(".")[:2])
    assert built[:2] == (major, minor)
