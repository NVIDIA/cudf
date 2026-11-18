# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudf_polars_cucascade import ColumnStats, ScanCache, ScanCacheConfig

LAYOUT = {
    "table": "s3://bucket/t",
    "columns": ["a", "b"],
    "column_of_chunk": [0, 1],
    "offsets": [0, 4000],
    "sizes": [4000, 4000],
}


def test_config_keyword_only():
    with pytest.raises(TypeError):
        ScanCacheConfig(256 << 20, "http://127.0.0.1:9", "us-east-1", "k", "s")


def test_config_requires_credentials():
    with pytest.raises(TypeError):
        ScanCacheConfig(cache_bytes=1 << 20)


def test_scan_cache_requires_size(make_config):
    with pytest.raises(ValueError, match="cache_bytes"):
        ScanCache(make_config(cache_bytes=0))


@pytest.mark.parametrize("numa_node", [None, 0, -1])
def test_scan_cache_numa_node(make_config, numa_node):
    assert (
        ScanCache(make_config(numa_node=numa_node))
        .summary()
        .startswith("fs_cache:")
    )


def test_scan_cache_requires_config():
    with pytest.raises(TypeError):
        ScanCache(None)


def test_register_layout_stats(scan_cache):
    scan_cache.register_layout("s3://bucket/t/f1.parquet", **LAYOUT)
    stats = {s.name: s for s in scan_cache.column_stats()}
    assert all(isinstance(s, ColumnStats) for s in stats.values())
    assert stats["s3://bucket/t/a"].registered == 4000
    assert stats["s3://bucket/t/b"].registered == 4000
    assert stats["s3://bucket/t/a"].requested == 0


def test_register_layout_idempotent(scan_cache):
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
def test_register_layout_malformed(scan_cache, overrides, match):
    with pytest.raises(ValueError, match=match):
        scan_cache.register_layout(
            "s3://bucket/t/bad.parquet", **{**LAYOUT, **overrides}
        )


def test_summary(scan_cache):
    assert scan_cache.summary().startswith("fs_cache:")


def test_open_unsupported_path(scan_cache):
    with pytest.raises(ValueError):
        scan_cache.open("ftp://bucket/a.parquet")
