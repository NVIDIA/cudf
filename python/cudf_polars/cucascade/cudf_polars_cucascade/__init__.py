# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A host-memory cache for the remote parquet reads of cudf-polars, built on cuCascade.

This is an optional extension that is built from source, against the libcudf and pylibcudf of
the environment it runs in. See ``README.md`` in the directory above this package.
"""

from cudf_polars_cucascade import (
    advising_datasource,
    build_info,
    column_admission,
    scan_cache,
)
from cudf_polars_cucascade.advising_datasource import AdvisingDatasource
from cudf_polars_cucascade.build_info import built_against_cudf_version
from cudf_polars_cucascade.column_admission import ColumnStats
from cudf_polars_cucascade.scan_cache import ScanCache, ScanCacheConfig

__all__ = [
    "AdvisingDatasource",
    "ColumnStats",
    "ScanCache",
    "ScanCacheConfig",
    "advising_datasource",
    "build_info",
    "built_against_cudf_version",
    "column_admission",
    "scan_cache",
]
