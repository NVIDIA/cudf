# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking scan statistics."""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import DataFrameScan, Scan
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.io import _build_source_info

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars


@nvtx_annotate_cudf_polars(message="collect_statistics")
def collect_statistics(
    root: IR,
    config_options: ConfigOptions[StreamingExecutor],
    executor: concurrent.futures.Executor,
) -> StatsCollector:
    """
    Collect DataSourceInfo for each leaf Scan/DataFrameScan node.

    Parameters
    ----------
    root: IR
        The root of the IR graph.
    config_options: ConfigOptions[StreamingExecutor]
        The configuration options.
    executor: concurrent.futures.Executor
        Executor to use for IO operations. This function does not start
        or shutdown the executor.
    """
    # Group parquet Scan nodes by paths, accumulating the union of needed columns
    # across all Scan nodes that read the same files.
    parquet_groups: dict[tuple[str, ...], tuple[set[str], Schema, list[Scan]]] = {}
    dataframe_scans: list[DataFrameScan] = []
    for node in traversal([root]):
        if isinstance(node, Scan):
            if node.typ == "parquet":
                paths_key = tuple(node.paths)
                if paths_key not in parquet_groups:
                    parquet_groups[paths_key] = (set(), {}, [])
                needed_cols, schema, scan_nodes = parquet_groups[paths_key]
                needed_cols.update(node.schema.keys())
                schema.update(node.schema)
                scan_nodes.append(node)
        elif isinstance(node, DataFrameScan):
            dataframe_scans.append(node)

    stats = StatsCollector()

    future_to_scan_nodes = {
        executor.submit(
            _build_source_info,
            scan_nodes[0],
            config_options,
            needed_cols=frozenset(needed_cols),
            schema=tuple(schema.items()),
        ): scan_nodes
        for needed_cols, schema, scan_nodes in parquet_groups.values()
    }

    try:
        for future in concurrent.futures.as_completed(future_to_scan_nodes):
            scan_nodes = future_to_scan_nodes[future]
            source = future.result()
            for node in scan_nodes:
                stats.scan_stats[node] = source
    except Exception:
        for pending in future_to_scan_nodes:
            pending.cancel()
        raise

    # DataFrameScan sources
    for node in dataframe_scans:
        stats.scan_stats[node] = _build_source_info(node, config_options)

    return stats
