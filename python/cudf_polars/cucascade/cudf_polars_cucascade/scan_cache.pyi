# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from cudf_polars_cucascade.advising_datasource import AdvisingDatasource
from cudf_polars_cucascade.column_admission import ColumnStats

__all__ = ["ScanCache", "ScanCacheConfig"]

class ScanCacheConfig:
    def __init__(
        self,
        *,
        cache_bytes: int,
        endpoint: str,
        region: str,
        access_key: str,
        secret_key: str,
        session_token: str = "",
        block_bytes: int | None = None,
        eviction_threshold: float | None = None,
        eviction_batch: float | None = None,
        admission_budget: float | None = None,
        min_advise_bytes: int | None = None,
        reactors: int | None = None,
        connections: int | None = None,
        stall_seconds: int | None = None,
        pinning_threads: int | None = None,
        numa_node: int | None = None,
    ) -> None: ...

class ScanCache:
    def __init__(self, config: ScanCacheConfig) -> None: ...
    def open(self, path: str) -> AdvisingDatasource: ...
    def register_layout(
        self,
        path: str,
        table: str,
        columns: Sequence[str],
        column_of_chunk: Sequence[int],
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> None: ...
    def column_stats(self) -> list[ColumnStats]: ...
    def summary(self) -> str: ...
