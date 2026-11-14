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
        block_bytes: int = 262144,
        eviction_threshold: float = 0.97,
        eviction_batch: float = 0.02,
        admission_budget: float = 0.96,
        min_advise_bytes: int = 262144,
        reactors: int = 16,
        connections: int = 128,
        stall_seconds: int = 5,
        pinning_threads: int = 8,
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
