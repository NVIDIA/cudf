# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for IR nodes."""

from __future__ import annotations

import concurrent.futures
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.io import Scan, StreamingScan

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import StatsCollector


@dataclass(frozen=True)
class CachedParquetInfo:
    """
    Metadata for a parquet file.

    File metadata is only cached when the setting
    ``ParquetOptions.prefetch_file_metadata`` is ``True``. Metadata is cached
    for the duration of the query.

    Parameters
    ----------
    path
        The path of an individual parquet file. This is one element of a
        ``paths`` tuple in a ``Scan`` node.
    size
        The size of the parquet file, in bytes. This is typically only set
        for remote URLs, since it allows skipping subsequent HTTP HEAD requests
        made by kvikio on operations involving that file.
    file_metadata
        The ``FileMetaData`` object for the parquet file returned from
        ``read_parquet_footers``.
    """

    path: str
    size: int | None
    file_metadata: plc.io.parquet_metadata.FileMetaData


@nvtx_annotate_cudf_polars(message="fetch_parquet_footers_for_paths")
def _prefetch_parquet_footers_for_paths(paths: list[str]) -> list[CachedParquetInfo]:
    """
    Prefetch parquet footers for a list of paths.

    This is typically executed concurrently with prefetch operations for other
    path groups for other parquet scan nodes.

    Parameters
    ----------
    paths
        The paths to prefetch.

    Returns
    -------
    paths
        The original input ``paths``.
    metadata
        The list of ``FileMetaData`` objects for the ``paths``.
    """
    # TODO: https://github.com/rapidsai/cudf/issues/22734, use object metadata from polars
    # For now, we'll just use kvikio to explicitly get the size.
    sizes: list[int | None] = []

    try:  # pragma: no cover; kvikio is optional
        import kvikio
    except ImportError:
        kvikio = None

    for path in paths:
        if (
            paths and kvikio is not None and plc.io.SourceInfo._is_remote_uri(path)
        ):  # pragma: no cover; kvikio is optional
            # We're OK to use `kvikio.RemoteFile.open` here. It does make an HTTP HEAD
            # request for S3/HTTP endpoints, but that's the entire reason we're running
            # this code. So long as it makes just *one* HTTP request, there's no advantage
            # to inferring the endpoint type.
            with kvikio.RemoteFile.open(path) as remote_file:
                sizes.append(remote_file.nbytes())
        else:
            sizes.append(None)

    metadata = plc.io.parquet_metadata.read_parquet_footers(
        plc.io.types.SourceInfo(
            [
                plc.io.types.FilepathSource(path, size)
                for path, size in zip(paths, sizes, strict=True)
            ]
        )
    )

    return [
        CachedParquetInfo(path, size, file_metadata)
        for path, size, file_metadata in zip(paths, sizes, metadata, strict=True)
    ]


@nvtx_annotate_cudf_polars(message="prefetch_parquet_file_metadata_for_ir")
def prefetch_parquet_file_metadata_for_ir(
    root: IR,
    py_executor: concurrent.futures.Executor | None,
    stats: StatsCollector | None = None,
) -> dict[str, CachedParquetInfo]:
    """
    Prefetch parquet metadata for all parquet scans in an IR graph.

    Parameters
    ----------
    root
        The root of the IR graph, which will be traversed.
    py_executor
        The thread pool executor to use for fetching parquet metadata concurrently.
    stats
        The stats collector. The file metadata might have already been
        prefetched during statistics collection, when the number of files
        sampled equals the total number of files. Providing ``stats`` here will
        skip rereading metadata for those files.

    Returns
    -------
    A dictionary mapping each individual path to its cached parquet metadata.
    """
    from cudf_polars.streaming.io import ParquetSourceInfo, StreamingScan

    all_paths: set[str] = set()

    for node in traversal([root]):
        if isinstance(node, StreamingScan):
            for scan in node.scans:
                for path in scan.paths:
                    all_paths.add(path)
        elif isinstance(node, Scan) and node.typ == "parquet":  # pragma: no cover
            raise RuntimeError("Unexpected parquet 'Scan' node in lowered IR graph.")

    cached_parquet_info: dict[str, CachedParquetInfo] = {}
    if stats is not None:
        for node, datasource_info in stats.scan_stats.items():
            if (
                isinstance(node, Scan)
                and node.typ == "parquet"
                and isinstance(datasource_info, ParquetSourceInfo)
                and datasource_info.cached_parquet_info is not None
            ):
                for info in datasource_info.cached_parquet_info:
                    cached_parquet_info[info.path] = info

    missing_paths = all_paths - set(cached_parquet_info.keys())
    cm: contextlib.AbstractContextManager[concurrent.futures.Executor | None]

    if py_executor is None:
        cm = py_executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="cudf-polars-io"
        )
    else:
        # We didn't create the executor, so we don't close it.
        cm = contextlib.nullcontext()

    with cm:
        futures = [
            py_executor.submit(_prefetch_parquet_footers_for_paths, [path])
            for path in missing_paths
        ]

        for future in concurrent.futures.as_completed(futures):
            for info in future.result():
                cached_parquet_info[info.path] = info
    return cached_parquet_info


def attach_cached_parquet_metadata(
    root: IR,
    cached_parquet_info_map: dict[str, CachedParquetInfo],
) -> None:
    """
    Attach prefetched metadata to scan nodes.

    This is an optimization only and does not affect IR identity.

    Parameters
    ----------
    root
        Root of the IR graph to update.
    cached_parquet_info_map
        Mapping from file paths to cached parquet metadata.
    """
    for node in traversal([root]):
        if isinstance(node, StreamingScan):
            for scan in node.scans:
                cached = [cached_parquet_info_map[path] for path in scan.paths]
                Scan._validate_cached_parquet_info(scan.paths, cached)
                scan.cached_parquet_info = cached
                scan._non_child_args = (*scan._non_child_args[:-1], cached)
