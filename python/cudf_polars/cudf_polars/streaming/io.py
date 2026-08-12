# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import polars as pl

import pylibcudf as plc

from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Empty,
    PythonScan,
    Scan,
    Sink,
)
from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.streaming.base import (
    IOPartitionFlavor,
    IOPartitionPlan,
    PartitionInfo,
    SerializedDataSourceInfo,
)
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.utils.config import Cluster
from cudf_polars.utils.cuda_stream import get_cuda_stream
from cudf_polars.utils.versions import POLARS_VERSION_LT_137

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping, Sequence

    from cudf_polars.containers import DataFrame, DataType
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import CachedParquetInfo, IRExecutionContext
    from cudf_polars.streaming.base import (
        DataSourceInfo,
        SerializedDataSourceInfo,
        StatsCollector,
    )
    from cudf_polars.streaming.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import (
        ConfigOptions,
        ParquetOptions,
        StreamingExecutor,
    )


@lower_ir_node.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]

    # NOTE: We calculate the expected partition count
    # to help trigger fallback warnings in lower_ir_graph.
    # The generate_ir_sub_network logic is NOT required
    # to obey this partition count. However, the count
    # WILL match after an IO operation (for now).
    rows_per_partition = config_options.executor.max_rows_per_partition
    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)

    return ir, {ir: PartitionInfo(count=count)}


def scan_partition_plan(
    ir: Scan, stats: StatsCollector, config_options: ConfigOptions[StreamingExecutor]
) -> IOPartitionPlan:
    """Extract the partitioning plan of a Scan operation."""
    if ir.typ == "parquet":
        blocksize: int = config_options.executor.target_partition_size
        if source := stats.scan_stats.get(ir):
            column_sizes = [
                sz
                for col in ir.schema
                if (sz := source.column_storage_size(col)) is not None
            ]
            if (file_size := sum(column_sizes)) > 0:
                if file_size > blocksize:
                    k_lo = file_size // blocksize
                    k_hi = k_lo + 1
                    factor = (
                        k_lo
                        if abs(file_size / k_lo - blocksize)
                        <= abs(file_size / k_hi - blocksize)
                        else k_hi
                    )
                    if factor >= 2:
                        return IOPartitionPlan(
                            factor,
                            IOPartitionFlavor.SPLIT_FILES,
                            estimated_chunk_bytes=file_size // factor,
                        )
                else:
                    k_lo = min(blocksize // int(file_size), len(ir.paths))
                    k_hi = k_lo + 1
                    factor = (
                        k_hi
                        if k_hi <= len(ir.paths)
                        and abs(k_hi * file_size - blocksize)
                        <= abs(k_lo * file_size - blocksize)
                        else k_lo
                    )
                return IOPartitionPlan(
                    factor,
                    IOPartitionFlavor.FUSED_FILES,
                    estimated_chunk_bytes=file_size * factor,
                )

    # TODO: Use file sizes for csv and json
    return IOPartitionPlan(1, IOPartitionFlavor.SINGLE_FILE)


def _rank_slice(total: int, rank: int, nranks: int) -> tuple[int, int]:
    """Return the partition range owned by this rank."""
    count = math.ceil(total / nranks)
    return count * rank, count


def expand_scan_for_rank(
    ir: Scan,
    plan: IOPartitionPlan,
    partition_count: int,
    *,
    rank: int,
    nranks: int,
    parquet_options: ParquetOptions,
) -> StreamingScan:
    """
    Expand a Scan node into a rank-local StreamingScan.

    Parameters
    ----------
    ir
        The Scan node to expand.
    plan
        The IO partitioning plan for the scan.
    partition_count
        Total number of partitions across all ranks.
    rank
        Rank of the current worker.
    nranks
        Number of workers.
    parquet_options
        Parquet reader options.

    Returns
    -------
    StreamingScan
        Rank-local streaming scan.
    """
    if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
        return StreamingScan.for_split_files(
            ir,
            plan,
            partition_count,
            rank=rank,
            nranks=nranks,
            parquet_options=parquet_options,
        )
    else:
        return StreamingScan.for_fused_files(
            ir,
            plan,
            partition_count,
            rank=rank,
            nranks=nranks,
            parquet_options=parquet_options,
        )


class SplitScan(IR):
    """
    Input from a split file.

    This class wraps a single-file ``Scan`` object. At
    IO/evaluation time, this class will only perform
    a partial read of the underlying file. The range
    (skip_rows and n_rows) is calculated at IO time.
    """

    __slots__ = (
        "base_scan",
        "cached_parquet_info",
        "parquet_options",
        "paths",
        "schema",
        "split_index",
        "total_splits",
    )
    _non_child = (
        "schema",
        "base_scan",
        "paths",
        "split_index",
        "total_splits",
        "parquet_options",
    )
    _n_non_child_args = 13
    base_scan: Scan
    """Scan operation this node is based on."""
    paths: list[str]
    """File path for this split task."""
    split_index: int
    """Index of the current split."""
    total_splits: int
    """Total number of splits."""
    parquet_options: ParquetOptions
    """Parquet-specific options."""
    cached_parquet_info: list[CachedParquetInfo] | None

    def __init__(
        self,
        schema: Schema,
        base_scan: Scan,
        paths: list[str],
        split_index: int,
        total_splits: int,
        parquet_options: ParquetOptions,
        cached_parquet_info: list[CachedParquetInfo] | None = None,
    ):
        self.schema = schema
        self.base_scan = base_scan
        self.paths = paths
        self.split_index = split_index
        self.total_splits = total_splits
        self._non_child_args = (
            split_index,
            total_splits,
            base_scan.schema,
            base_scan.typ,
            base_scan.reader_options,
            paths,
            base_scan.with_columns,
            base_scan.skip_rows,
            base_scan.n_rows,
            base_scan.row_index,
            base_scan.include_file_paths,
            base_scan.predicate,
            parquet_options,
            cached_parquet_info,
        )
        self.parquet_options = parquet_options
        self.cached_parquet_info = cached_parquet_info
        self.children = ()
        if base_scan.typ not in ("parquet",):  # pragma: no cover
            raise NotImplementedError(
                f"Unhandled Scan type for file splitting: {base_scan.typ}"
            )

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (
            type(self),
            tuple(self.schema.items()),
            self.base_scan.get_hashable(),
            tuple(self.paths),
            self.split_index,
            self.total_splits,
            self.parquet_options,
        )

    @classmethod
    def do_evaluate(
        cls,
        split_index: int,
        total_splits: int,
        schema: Schema,
        typ: str,
        reader_options: dict[str, Any],
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
        predicate: NamedExpr | None,
        parquet_options: ParquetOptions,
        cached_parquet_info: list[CachedParquetInfo] | None,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        if typ not in ("parquet",):  # pragma: no cover
            raise NotImplementedError(f"Unhandled Scan type for file splitting: {typ}")

        if len(paths) > 1:  # pragma: no cover
            raise ValueError(f"Expected a single path, got: {paths}")

        # Parquet logic:
        # - We are one of "total_splits" SplitScan nodes
        #   assigned to the same file.
        # - We know our index within this file ("split_index")
        # - We can also use parquet metadata to query the
        #   total number of rows in each row-group of the file.
        # - We can use all this information to calculate the
        #   "skip_rows" and "n_rows" options to use locally.

        if cached_parquet_info is not None:
            parquet_metadatas = [info.file_metadata for info in cached_parquet_info]

            row_group_num_rows = [
                num_rows
                for metadata in parquet_metadatas
                for num_rows in metadata.row_group_num_rows
            ]

        else:
            row_group_num_rows = [
                rg["num_rows"]
                for rg in plc.io.parquet_metadata.read_parquet_metadata(
                    plc.io.SourceInfo(paths)
                ).rowgroup_metadata()
            ]

        total_row_groups = len(row_group_num_rows)
        if total_splits <= total_row_groups:
            # We have enough row-groups in the file to align
            # all "total_splits" of our reads with row-group
            # boundaries. Calculate which row-groups to include
            # in the current read, and use metadata to translate
            # the row-group indices to "skip_rows" and "n_rows".
            rg_stride = total_row_groups // total_splits
            skip_rgs = rg_stride * split_index
            skip_rows = sum(row_group_num_rows[:skip_rgs])
            n_rows = sum(row_group_num_rows[skip_rgs : skip_rgs + rg_stride])
        else:
            # There are not enough row-groups to align
            # all "total_splits" of our reads with row-group
            # boundaries. Use metadata to directly calculate
            # "skip_rows" and "n_rows" for the current read.
            total_rows = sum(row_group_num_rows)
            n_rows = total_rows // total_splits
            skip_rows = n_rows * split_index

        # Last split should always read to end of file
        if split_index == (total_splits - 1):
            n_rows = -1

        # Perform the partial read
        with nvtx_annotate_cudf_polars(
            message=f"SplitScan: {paths[0]} [{split_index + 1}/{total_splits}]"
        ):
            return Scan.do_evaluate(
                schema,
                typ,
                reader_options,
                paths,
                with_columns,
                skip_rows,
                n_rows,
                row_index,
                include_file_paths,
                predicate,
                parquet_options,
                cached_parquet_info,
                context=context,
            )


class FusedScan(IR):
    """
    Input from one or more complete files read as a single task.

    Covers both FUSED_FILES (N > 1 small files grouped together) and
    SINGLE_FILE (N = 1).
    """

    __slots__ = (
        "base_scan",
        "cached_parquet_info",
        "parquet_options",
        "paths",
        "schema",
    )
    _non_child = (
        "schema",
        "base_scan",
        "paths",
        "parquet_options",
    )
    _n_non_child_args = 11
    base_scan: Scan
    """Scan operation this node is based on."""
    paths: list[str]
    """File paths assigned to this task."""
    parquet_options: ParquetOptions
    """Parquet-specific options."""
    cached_parquet_info: list[CachedParquetInfo] | None
    """Cached parquet metadata."""

    def __init__(
        self,
        schema: Schema,
        base_scan: Scan,
        paths: list[str],
        parquet_options: ParquetOptions,
        cached_parquet_info: list[CachedParquetInfo] | None = None,
    ):
        self.schema = schema
        self.base_scan = base_scan
        self.paths = paths
        self.parquet_options = parquet_options
        self.cached_parquet_info = cached_parquet_info
        self._non_child_args = (
            base_scan.schema,
            base_scan.typ,
            base_scan.reader_options,
            paths,
            base_scan.with_columns,
            base_scan.skip_rows,
            base_scan.n_rows,
            base_scan.row_index,
            base_scan.include_file_paths,
            base_scan.predicate,
            parquet_options,
            cached_parquet_info,
        )
        self.children = ()

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (
            type(self),
            tuple(self.schema.items()),
            self.base_scan.get_hashable(),
            tuple(self.paths),
            self.parquet_options,
        )

    @classmethod
    def do_evaluate(
        cls,
        schema: Schema,
        typ: str,
        reader_options: dict[str, Any],
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
        predicate: NamedExpr | None,
        parquet_options: ParquetOptions,
        cached_parquet_info: list[CachedParquetInfo] | None,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        with nvtx_annotate_cudf_polars(message=f"FusedScan: {', '.join(paths)}"):
            return Scan.do_evaluate(
                schema,
                typ,
                reader_options,
                paths,
                with_columns,
                skip_rows,
                n_rows,
                row_index,
                include_file_paths,
                predicate,
                parquet_options,
                cached_parquet_info,
                context=context,
            )


@lower_ir_node.register(Empty)
def _(
    ir: Empty, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    return ir, {ir: PartitionInfo(count=1)}  # pragma: no cover


@lower_ir_node.register(PythonScan)
def _(
    ir: PythonScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # A PythonScan can emit multiple chunks and the count is unknown at lowering,
    # so it always lowers as multi-partition (count > 1) regardless of world size.
    # This forces downstream global operators to insert the required reduction
    # instead of evaluating chunkwise.
    #
    # TODO: Remove this workaround once dynamic planning is mandatory. Under
    # dynamic planning the runtime adapts to the real chunk count, so this
    # lowering estimate no longer gates correctness.
    return ir, {ir: PartitionInfo(count=2)}


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[StreamingScan, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]
    parquet_options = config_options.parquet_options
    if (
        ir.typ in ("csv", "parquet", "ndjson")
        and ir.n_rows == -1
        and ir.skip_rows == 0
        and ir.row_index is None
    ):
        # NOTE: We calculate the expected partition count
        # to help trigger fallback warnings in lower_ir_graph.
        # The generate_ir_sub_network logic is NOT required
        # to obey this partition count. However, the count
        # WILL match after an IO operation (for now).
        plan = scan_partition_plan(ir, rec.state["stats"], config_options)
        paths = list(ir.paths)
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
            count = plan.factor * len(paths)
        else:
            count = math.ceil(len(paths) / plan.factor)
    else:
        plan = IOPartitionPlan(
            flavor=IOPartitionFlavor.SINGLE_READ, factor=len(ir.paths)
        )
        count = 1

    parquet_options = dataclasses.replace(parquet_options, chunked=False)

    new_ir = expand_scan_for_rank(
        ir,
        plan,
        count,
        rank=rec.state["rank"],
        nranks=rec.state["nranks"],
        parquet_options=parquet_options,
    )
    return new_ir, {new_ir: PartitionInfo(count=count, io_plan=plan)}


class StreamingScan(IR):
    """A streaming scan node."""

    __slots__ = (
        "base_scan",
        "scan_type",
        "scans",
        "schema",
    )
    _non_child = (
        "scans",
        "base_scan",
        "scan_type",
    )
    _n_non_child_args = 3
    scans: Sequence[SplitScan] | Sequence[FusedScan]
    base_scan: Scan

    def __init__(
        self,
        scans: Sequence[SplitScan] | Sequence[FusedScan],
        base_scan: Scan,
        scan_type: Literal["split", "fused"],
    ):
        self.scans = scans
        self.base_scan = base_scan
        self.schema = base_scan.schema
        self.scan_type = scan_type
        self._non_child_args = (scans, base_scan, scan_type)
        self.children = ()

    @classmethod
    def for_split_files(
        cls,
        base_scan: Scan,
        plan: IOPartitionPlan,
        partition_count: int,
        *,
        rank: int,
        nranks: int,
        parquet_options: ParquetOptions,
    ) -> Self:
        """Construct a StreamingScan where each file is split into factor partitions."""
        local_offset, local_count = _rank_slice(partition_count, rank, nranks)
        path_offset = local_offset // plan.factor
        path_end = math.ceil((local_offset + local_count) / plan.factor)
        local_paths = base_scan.paths[path_offset:path_end]
        sindex = local_offset % plan.factor
        scans: list[SplitScan] = []
        splits_created = 0
        for path in local_paths:
            while sindex < plan.factor and splits_created < local_count:
                scans.append(
                    SplitScan(
                        base_scan.schema,
                        base_scan,
                        [path],
                        sindex,
                        plan.factor,
                        parquet_options,
                        None,
                    )
                )
                sindex += 1
                splits_created += 1
            sindex = 0
        return cls(scans, base_scan, "split")

    @classmethod
    def for_fused_files(
        cls,
        base_scan: Scan,
        plan: IOPartitionPlan,
        partition_count: int,
        *,
        rank: int,
        nranks: int,
        parquet_options: ParquetOptions,
    ) -> Self:
        """Construct a StreamingScan where factor files are grouped into one partition."""
        local_offset, local_count = _rank_slice(partition_count, rank, nranks)
        paths_start = local_offset * plan.factor
        paths_end = paths_start + plan.factor * local_count
        scans = [
            FusedScan(
                base_scan.schema,
                base_scan,
                base_scan.paths[offset : offset + plan.factor],
                parquet_options,
                None,
            )
            for offset in range(paths_start, paths_end, plan.factor)
            if base_scan.paths[offset : offset + plan.factor]
        ]
        return cls(scans, base_scan, "fused")

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        # We don't need to include base_scan / schema, since it's in all the scan nodes.
        return (type(self), *tuple(x.get_hashable() for x in self.scans))

    @classmethod
    def do_evaluate(
        cls,
        scans: Sequence[SplitScan] | Sequence[FusedScan],
        base_scan: Scan,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Raises NotImplementedError for StreamingScan nodes."""
        raise NotImplementedError(
            "StreamingScan.do_evaluate should not be called directly. Call Scan.do_evaluate on each scan node instead."
        )


class StreamingSink(IR):
    """Sink a dataframe in streaming mode."""

    __slots__ = ("sink", "sink_to_directory")
    _non_child = ("schema", "sink", "sink_to_directory")
    _n_non_child_args = 0

    sink: Sink
    sink_to_directory: bool

    def __init__(
        self,
        schema: Schema,
        sink: Sink,
        sink_to_directory: bool,  # noqa: FBT001
        df: IR,
    ) -> None:
        # Order must match ``_non_child`` + ``children`` so :meth:`Node.__reduce__`
        # / ``reconstruct`` round-trip over pickling (e.g. Dask workers).
        self.schema = schema
        self.sink = sink
        self.sink_to_directory = sink_to_directory
        self._non_child_args = ()
        self.children = (df,)

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (type(self), self.sink, *self.children)


@lower_ir_node.register(Sink)
def _(
    ir: Sink, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    executor_options = rec.state["config_options"].executor

    assert executor_options.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )

    # TODO: Support cloud storage
    if (
        Path(ir.path).exists()
        and executor_options.sink_to_directory
        and executor_options.cluster == Cluster.DEFAULT_SINGLETON
    ):
        # This lowering-time check can't be performed with the spmd / ray / dask
        # clusters, which lower on each worker independently. There's a race condition
        # between each worker performing this check that the path doesn't yet exist,
        # and the sink operation creating the directory at the start of execution.
        raise NotImplementedError(
            f"Trying to sink to an existing directory: {ir.path}. "
            "Writing to an existing path is not supported when sinking "
            "to a directory. Please remove the target directory before "
            "calling 'collect'."
        )

    sink_to_directory = executor_options.sink_to_directory
    assert sink_to_directory is not None  # set in StreamingExecutor.__post_init__
    new_node = StreamingSink(
        ir.schema,
        ir.reconstruct([child]),
        sink_to_directory,
        child,
    )
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info


def _prepare_sink_directory(path: str) -> None:
    """Prepare for a multi-partition sink."""
    # TODO: Support cloud storage
    Path(path).mkdir(parents=True, exist_ok=True)


def _sink_to_parquet_file(
    path: str,
    options: dict[str, Any],
    writer: plc.io.parquet.ChunkedParquetWriter | None,
    df: DataFrame,
) -> plc.io.parquet.ChunkedParquetWriter:
    """Sink a partition to an open Parquet file."""
    # Set up a new chunked Parquet writer if necessary.
    if writer is None:
        metadata = Sink._make_parquet_metadata(df)
        sink = plc.io.types.SinkInfo([path])
        builder = Sink._apply_parquet_writer_options(
            plc.io.parquet.ChunkedParquetWriterOptions.builder(sink), options
        )
        writer_options = builder.metadata(metadata).build()
        writer = plc.io.parquet.ChunkedParquetWriter.from_options(
            writer_options, stream=df.stream
        )

    # Append to the open Parquet file.
    assert isinstance(writer, plc.io.parquet.ChunkedParquetWriter), (
        "ChunkedParquetWriter is required."
    )
    writer.write(df.table)

    return writer


@overload
def _sink_to_file(
    kind: Literal["Parquet"],
    path: str,
    options: dict[str, Any],
    writer_state: plc.io.parquet.ChunkedParquetWriter,
    df: DataFrame,
) -> plc.io.parquet.ChunkedParquetWriter: ...


@overload
def _sink_to_file(
    kind: str,
    path: str,
    options: dict[str, Any],
    writer_state: None,
    df: DataFrame,
) -> Literal[True]: ...


def _sink_to_file(
    kind: str,
    path: str,
    options: dict[str, Any],
    writer_state: Any,
    df: DataFrame,
) -> Literal[True] | plc.io.parquet.ChunkedParquetWriter:
    """Sink a partition to an open file."""
    if kind == "Parquet":
        # Parquet writer will pass along a
        # ChunkedParquetWriter "writer state".
        return _sink_to_parquet_file(
            path,
            options,
            writer_state,
            df,
        )
    elif kind == "Csv":
        use_options = options.copy()
        if writer_state is None:
            mode = "wb"
        else:
            mode = "ab"
            use_options["include_header"] = False
        with Path.open(Path(path), mode) as f:
            # Path.open returns IO[Any] but SinkInfo needs more specific IO types
            sink = plc.io.types.SinkInfo([f])  # type: ignore[arg-type]
            Sink._write_csv(sink, use_options, df)
    elif kind == ("Json" if POLARS_VERSION_LT_137 else "NDJson"):
        mode = "wb" if writer_state is None else "ab"
        with Path.open(Path(path), mode) as f:
            # Path.open returns IO[Any] but SinkInfo needs more specific IO types
            sink = plc.io.types.SinkInfo([f])  # type: ignore[arg-type]
            Sink._write_json(sink, df)
    else:  # pragma: no cover; Shouldn't get here.
        raise NotImplementedError(f"{kind} not yet supported in _sink_to_file")

    return True


def _columnchunk_metadata_from_footers(
    footers: list[plc.io.parquet_metadata.FileMetaData],
) -> dict[str, list[int]]:
    columnchunk_metadata: dict[str, list[int]] = {}
    for fmd in footers:
        for name, uncompressed_sizes in fmd.columnchunk_metadata.items():
            columnchunk_metadata.setdefault(name, []).extend(uncompressed_sizes)
    return columnchunk_metadata


class ParquetMetadata:
    """
    Parquet metadata container.

    Parameters
    ----------
    paths
        Parquet-dataset paths.
    max_footer_samples
        Maximum number of file footers to sample metadata from.
    """

    __slots__ = (
        "cached_parquet_info",
        "column_names",
        "max_footer_samples",
        "mean_size_per_file",
        "num_row_groups_per_file",
        "paths",
        "row_count",
        "sample_paths",
        "sampled_file_count",
        "total_file_count",
    )

    paths: tuple[str, ...]
    """Parquet-dataset paths."""
    max_footer_samples: int
    """Maximum number of file footers to sample metadata from."""
    row_count: int | None
    """Total row-count estimate."""
    num_row_groups_per_file: tuple[int, ...]
    """Number of row groups in each sampled file."""
    mean_size_per_file: dict[str, int]
    """Average column storage size in a single file."""
    column_names: tuple[str, ...]
    """All column names found it the dataset."""
    sample_paths: tuple[str, ...]
    """Sampled file paths."""
    cached_parquet_info: list[CachedParquetInfo] | None
    """Cached parquet info for the sampled paths. Only set if all files were sampled."""

    @nvtx_annotate_cudf_polars(message="ParquetMetadata")
    def __init__(self, paths: tuple[str, ...], max_footer_samples: int):
        from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths

        self.paths = paths
        self.max_footer_samples = max_footer_samples
        self.row_count = None
        self.num_row_groups_per_file = ()
        self.mean_size_per_file = {}
        self.column_names = ()
        self.cached_parquet_info = None
        self.total_file_count = len(self.paths)
        self.sampled_file_count = 0
        if max_footer_samples <= 0:
            self.sample_paths = ()
            return

        stride = max(1, int(len(paths) / max_footer_samples))
        self.sample_paths = paths[: stride * max_footer_samples : stride]

        if not self.sample_paths:
            # No paths to sample from
            # TODO: This requires row_count to be nullable. Why do we allow empty paths?
            return

        sampled_file_count = len(self.sample_paths)

        sample_parquet_info = _prefetch_parquet_footers_for_paths(
            list(self.sample_paths)
        )
        sample_footers = [info.file_metadata for info in sample_parquet_info]

        sampled_row_count = sum(fmd.num_rows for fmd in sample_footers)
        if self.total_file_count == sampled_file_count:
            row_count = sampled_row_count
            self.cached_parquet_info = sample_parquet_info
        else:
            num_rows_per_sampled_file = int(sampled_row_count / sampled_file_count)
            row_count = num_rows_per_sampled_file * self.total_file_count

        num_row_groups_per_sampled_file = [
            len(fmd.row_group_num_rows) for fmd in sample_footers
        ]
        rowgroup_offsets_per_file = list(
            itertools.accumulate(num_row_groups_per_sampled_file, initial=0)
        )

        column_sizes_per_file = {
            name: [
                sum(uncompressed_sizes[start:end])
                for (start, end) in itertools.pairwise(rowgroup_offsets_per_file)
            ]
            for name, uncompressed_sizes in _columnchunk_metadata_from_footers(
                sample_footers
            ).items()
        }

        self.column_names = tuple(column_sizes_per_file)
        self.mean_size_per_file = {
            name: int(statistics.mean(sizes))
            for name, sizes in column_sizes_per_file.items()
        }
        self.num_row_groups_per_file = tuple(num_row_groups_per_sampled_file)
        self.row_count = row_count
        self.sampled_file_count = sampled_file_count


@nvtx_annotate_cudf_polars(message="_sample_rg_sizes")
def _sample_rg_sizes(
    metadata: ParquetMetadata,
    target_cols: list[str],
    max_row_group_samples: int,
) -> dict[str, int]:
    """Return mean uncompressed bytes per row-group for each column in target_cols."""
    sample_paths = metadata.sample_paths
    num_row_groups_per_file = metadata.num_row_groups_per_file
    if not sample_paths or len(num_row_groups_per_file) != len(sample_paths):
        return {}  # pragma: no cover

    n_sampled = 0
    samples: defaultdict[str, list[int]] = defaultdict(list)
    for path, num_rgs in zip(sample_paths, num_row_groups_per_file, strict=True):
        for rg_id in range(num_rgs):
            n_sampled += 1
            samples[path].append(rg_id)
            if n_sampled == max_row_group_samples:
                break
        if n_sampled == max_row_group_samples:
            break

    if not n_sampled:
        return {}  # pragma: no cover

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(list(samples))
    ).build()
    options.set_column_names(target_cols)
    options.set_row_groups(list(samples.values()))
    stream = get_cuda_stream()
    tbl_w_meta = plc.io.parquet.read_parquet(options, stream=stream)
    result = {
        name: column.device_buffer_size() // n_sampled
        for name, column in zip(
            tbl_w_meta.column_names(include_children=False),
            tbl_w_meta.columns,
            strict=True,
        )
    }
    stream.synchronize()
    return result


def _is_fixed_width(dtype: DataType) -> bool:
    """Return whether dtype is a concrete fixed-width type."""
    return dtype.id() not in (plc.TypeId.EMPTY, plc.TypeId.NUM_TYPE_IDS) and (
        plc.traits.is_fixed_width(dtype.plc_type)
    )


def _decoded_size_floor(dtype: DataType, nrows: int) -> int:
    """Return a conservative decoded-column byte floor for scan planning."""
    nullmask = (nrows + 7) // 8
    plc_dtype = dtype.plc_type
    if dtype.id() == plc.TypeId.STRING:
        # Decoded strings always have int32 offsets (4 bytes)
        return (nrows + 1) * 4 + nullmask
    if _is_fixed_width(dtype):
        return nrows * plc.types.size_of(plc_dtype) + nullmask
    return max(1, nrows)


class ParquetSourceInfo:
    """Parquet datasource information, fully computed at construction time."""

    type: Literal["parquet"] = "parquet"

    def __init__(
        self,
        row_count: int | None,
        per_file_means: dict[str, int] | None = None,
        *,
        # TODO: change this to cached_parquet_info
        cached_parquet_info: list[CachedParquetInfo] | None = None,
    ):
        if per_file_means is None:
            per_file_means = {}

        self.row_count = row_count
        self.per_file_means = per_file_means
        self.cached_parquet_info = cached_parquet_info

    @classmethod
    def from_paths(
        cls,
        paths: tuple[str, ...],
        needed_cols: frozenset[str],
        schema: tuple[tuple[str, DataType], ...],
        max_footer_samples: int,
        max_row_group_samples: int,
    ) -> ParquetSourceInfo:
        """Build a ParquetSourceInfo from a list of paths."""
        metadata = ParquetMetadata(paths, max_footer_samples)
        row_count = metadata.row_count

        file_count = len(paths)
        per_file_means: dict[str, int] = {}

        if not (file_count and row_count and needed_cols):
            return cls(row_count, {})

        rows_per_file = max(1, row_count // file_count)
        schema_map = dict(schema)
        sample_cols: list[str] = []

        for col in needed_cols:
            footer_mean = metadata.mean_size_per_file.get(col)
            if footer_mean is None:
                continue
            dtype = schema_map[col]
            decoded_floor = _decoded_size_floor(dtype, rows_per_file)
            # This is conservative for all-null columns; footer null counts could
            # refine the floor later if the extra partitioning becomes costly.
            if (
                footer_mean < decoded_floor
                and max_row_group_samples > 0
                and not _is_fixed_width(dtype)
            ):
                sample_cols.append(col)
            else:
                per_file_means[col] = max(footer_mean, decoded_floor)

        if sample_cols:
            rg_sizes = _sample_rg_sizes(metadata, sample_cols, max_row_group_samples)
            mean_rg_count = (
                statistics.mean(metadata.num_row_groups_per_file)
                if metadata.num_row_groups_per_file
                else 1
            )
            for col in sample_cols:
                rg_size = rg_sizes.get(col)
                decoded_floor = _decoded_size_floor(schema_map[col], rows_per_file)
                footer_mean = metadata.mean_size_per_file[col]
                per_file_means[col] = (
                    max(footer_mean, decoded_floor, int(rg_size * mean_rg_count))
                    if rg_size
                    else max(footer_mean, decoded_floor)
                )

        cached_parquet_info: list[CachedParquetInfo] | None
        if (
            metadata.sampled_file_count == metadata.total_file_count
            and metadata.cached_parquet_info is not None
        ):
            cached_parquet_info = list(metadata.cached_parquet_info)
        else:
            cached_parquet_info = None
        return cls(row_count, per_file_means, cached_parquet_info=cached_parquet_info)

    def column_storage_size(self, column: str) -> int | None:
        """Return the average storage size for a single column in one file."""
        return self.per_file_means.get(column)

    def serialize(self) -> SerializedDataSourceInfo:
        """Return JSON-serializable representation of the data source info."""
        return {
            "type": self.type,
            "row_count": self.row_count,
            "per_file_means": self.per_file_means,
        }

    @classmethod
    def deserialize(cls, data: SerializedDataSourceInfo) -> ParquetSourceInfo:
        """Deserialize a ParquetSourceInfo from a dictionary."""
        if data["type"] != "parquet":
            raise ValueError(f"Expected ParquetSourceInfo, got {data['type']}")
        return cls(data["row_count"], data["per_file_means"])


class DataFrameSourceInfo:
    """
    In-memory DataFrame source information.

    Parameters
    ----------
    row_count
        Exact row-count for the polars dataframe.
    """

    type: Literal["dataframe"] = "dataframe"

    def __init__(self, row_count: int):
        self.row_count = row_count

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> DataFrameSourceInfo:
        """Build a DataFrameSourceInfo from a polars dataframe."""
        return cls(df.height)

    def column_storage_size(self, column: str) -> int | None:
        """Return the average storage size for a single column in one file."""
        return None

    def serialize(self) -> SerializedDataSourceInfo:
        """Return JSON-serializable representation of the data source info."""
        return {
            "type": self.type,
            "row_count": self.row_count,
            "per_file_means": None,
        }

    @classmethod
    def deserialize(cls, data: SerializedDataSourceInfo) -> DataFrameSourceInfo:
        """Deserialize a DataFrameSourceInfo from a dictionary."""
        if data["type"] != "dataframe":
            raise ValueError(f"Expected DataFrameSourceInfo, got {data['type']}")
        if data["row_count"] is None:
            raise ValueError("Row count is required for DataFrameSourceInfo")
        return cls(data["row_count"])


@functools.cache
def _build_parquet_source(
    paths: tuple[str, ...],
    needed_cols: frozenset[str],
    schema: tuple[tuple[str, DataType], ...],
    max_footer_samples: int,
    max_row_group_samples: int,
) -> ParquetSourceInfo:
    """Return cached, fully-computed Parquet datasource information."""
    return ParquetSourceInfo.from_paths(
        paths, needed_cols, schema, max_footer_samples, max_row_group_samples
    )


def _build_source_info(
    ir: Scan | DataFrameScan,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    needed_cols: frozenset[str] | None = None,
    schema: tuple[tuple[str, DataType], ...] | None = None,
) -> DataSourceInfo:
    """Return DataSourceInfo for a Scan or DataFrameScan node."""
    if isinstance(ir, DataFrameScan):
        return DataFrameSourceInfo.from_polars(pl.DataFrame._from_pydf(ir.df))
    elif isinstance(ir, Scan) and ir.typ == "parquet":
        max_footer = config_options.parquet_options.max_footer_samples
        max_rg = config_options.parquet_options.max_row_group_samples
        needed_cols = frozenset(ir.schema) if needed_cols is None else needed_cols
        schema = tuple(ir.schema.items()) if schema is None else schema
        paths = tuple(ir.paths)
        return _build_parquet_source(paths, needed_cols, schema, max_footer, max_rg)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Scan type: {ir.typ}")


def _clear_source_info_cache() -> None:
    """Clear DataSourceInfo caches."""
    # TODO: Avoid clearing the cache if we can
    # check that the underlying data hasn't changed.
    _build_parquet_source.cache_clear()
