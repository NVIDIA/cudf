# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A host-memory cache for the remote parquet reads of cudf-polars, built on cuCascade.

This is an optional extension that is built from source, against the libcudf and pylibcudf of
the environment it runs in. See ``README.md`` in the directory above this package.
"""

from cudf_polars_cucascade._cucascade import (
    AdvisingDatasource,
    ColumnStats,
    ScanCache,
    ScanCacheOptions,
    built_against_cudf_version,
)

__all__ = [
    "AdvisingDatasource",
    "ColumnStats",
    "ScanCache",
    "ScanCacheOptions",
    "built_against_cudf_version",
]
