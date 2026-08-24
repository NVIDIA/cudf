# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

import enum
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import Generator

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR


class PartitionInfo:
    """Partitioning information."""

    __slots__ = ("count", "io_plan", "partitioned_on")
    count: int
    """Partition count."""
    partitioned_on: tuple[NamedExpr, ...]
    """Columns the data is hash-partitioned on."""
    io_plan: IOPartitionPlan | None
    """IO partitioning plan (Scan nodes only)."""

    def __init__(
        self,
        count: int,
        *,
        partitioned_on: tuple[NamedExpr, ...] = (),
        io_plan: IOPartitionPlan | None = None,
    ):
        self.count = count
        self.partitioned_on = partitioned_on
        self.io_plan = io_plan

    def __rich_repr__(self) -> Generator[Any, None, None]:
        """Formatting for rich.pretty.pprint."""
        yield "count", self.count
        yield "partitioned_on", self.partitioned_on


class SerializedDataSourceInfo(TypedDict):
    """The serialized form of DataSourceInfo."""

    type: Literal["parquet", "dataframe"]
    row_count: int | None
    per_file_means: dict[str, int] | None


class SerializedStatsEntry(TypedDict):
    """The serialized form of a stats entry."""

    index: int
    info: SerializedDataSourceInfo


class DataSourceInfo(Protocol):
    """
    Table data source information.

    Notes
    -----
    Sub-class for specific data source types (e.g. Parquet, DataFrame).
    """

    @property
    def type(self) -> Literal["parquet", "dataframe"]:
        """The type of the data source. Useful for serialization and deserialization."""

    @property
    def row_count(self) -> int | None:
        """Data source row-count estimate."""

    def column_storage_size(self, column: str) -> int | None:
        """Return the average storage size for a single column in one file."""

    def serialize(self) -> SerializedDataSourceInfo:
        """Return JSON-serializable representation of the data source info."""

    @classmethod
    def deserialize(cls, data: SerializedDataSourceInfo) -> DataSourceInfo:
        """Deserialize a DataSourceInfo from a dictionary."""


class StatsCollector:
    """Scan statistics collector."""

    __slots__ = ("scan_stats",)

    scan_stats: dict[IR, DataSourceInfo]
    """DataSourceInfo for each leaf Scan/DataFrameScan node."""

    def __init__(self) -> None:
        self.scan_stats: dict[IR, DataSourceInfo] = {}

    def serialize(self, ir: IR) -> list[SerializedStatsEntry]:
        """
        Serialize to a JSON-compatible list.

        IR nodes are represented by their position in a deterministic
        traversal of *ir* so that the result is independent of object
        identity.
        """
        node_to_idx = {node: i for i, node in enumerate(traversal([ir]))}
        return [
            {"index": node_to_idx[node], "info": info.serialize()}
            for node, info in self.scan_stats.items()
        ]

    @classmethod
    def deserialize(cls, entries: list[SerializedStatsEntry], ir: IR) -> StatsCollector:
        """
        Reconstruct a :class:`StatsCollector` from its serialized form.

        Parameters
        ----------
        entries
            Serialized stats produced by :meth:`serialize`.
        ir
            Root of the (pre-lowered) IR graph on the local rank.
        """
        from cudf_polars.streaming.io import DataFrameSourceInfo, ParquetSourceInfo

        _deserializers: dict[
            str, type[ParquetSourceInfo] | type[DataFrameSourceInfo]
        ] = {
            "parquet": ParquetSourceInfo,
            "dataframe": DataFrameSourceInfo,
        }
        idx_to_node = dict(enumerate(traversal([ir])))
        stats = cls()
        for entry in entries:
            info_data = entry["info"]
            info_cls = _deserializers[info_data["type"]]
            stats.scan_stats[idx_to_node[entry["index"]]] = info_cls.deserialize(
                info_data
            )
        return stats


class IOPartitionFlavor(IntEnum):
    """Flavor of IO partitioning."""

    SINGLE_FILE = enum.auto()  # 1:1 mapping between files and partitions
    SPLIT_FILES = enum.auto()  # Split each file into >1 partition
    FUSED_FILES = enum.auto()  # Fuse multiple files into each partition
    SINGLE_READ = enum.auto()  # One worker/task reads everything


class IOPartitionPlan:
    """
    IO partitioning plan.

    Notes
    -----
    The meaning of `factor` depends on the value of `flavor`:
      - SINGLE_FILE: `factor` must be `1`.
      - SPLIT_FILES: `factor` is the number of partitions per file.
      - FUSED_FILES: `factor` is the number of files per partition.
      - SINGLE_READ: `factor` is the total number of files.
    """

    __slots__ = ("estimated_chunk_bytes", "factor", "flavor")
    factor: int
    flavor: IOPartitionFlavor
    estimated_chunk_bytes: int | None
    """Estimated decompressed bytes per chunk (Scan nodes only)."""

    def __init__(
        self,
        factor: int,
        flavor: IOPartitionFlavor,
        *,
        estimated_chunk_bytes: int | None = None,
    ) -> None:
        if flavor == IOPartitionFlavor.SINGLE_FILE and factor != 1:  # pragma: no cover
            raise ValueError(f"Expected factor == 1 for {flavor}, got: {factor}")
        self.factor = factor
        self.flavor = flavor
        self.estimated_chunk_bytes = estimated_chunk_bytes
