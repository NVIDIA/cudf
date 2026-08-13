# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typing utilities for cudf_polars."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, MutableMapping, Sized
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NewType,
    Protocol,
    TypeVar,
    TypedDict,
    Union,
)

import polars as pl
import polars.datatypes
from polars import polars as plrs  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    import pylibcudf as plc

    from cudf_polars.containers import DataFrame, DataType
    from cudf_polars.dsl import nodebase


__all__: list[str] = [
    "ClosedInterval",
    "CollectKwargs",
    "ColumnHeader",
    "ColumnOptions",
    "DataFrameHeader",
    "GenericTransformer",
    "NodeTraverser",
    "OptimizationArgs",
    "PolarsExpr",
    "PolarsIR",
    "RankMethod",
    "Schema",
    "SizedIterator",
    "Slice",
]

PolarsIR: TypeAlias = Union[
    plrs._ir_nodes.PythonScan,
    plrs._ir_nodes.Scan,
    plrs._ir_nodes.Cache,
    plrs._ir_nodes.DataFrameScan,
    plrs._ir_nodes.Select,
    plrs._ir_nodes.GroupBy,
    plrs._ir_nodes.Join,
    plrs._ir_nodes.HStack,
    plrs._ir_nodes.Distinct,
    plrs._ir_nodes.Sort,
    plrs._ir_nodes.Slice,
    plrs._ir_nodes.Filter,
    plrs._ir_nodes.SimpleProjection,
    plrs._ir_nodes.MapFunction,
    plrs._ir_nodes.Union,
    plrs._ir_nodes.HConcat,
    plrs._ir_nodes.ExtContext,
]

PolarsExpr: TypeAlias = Union[
    plrs._expr_nodes.Function,
    plrs._expr_nodes.Window,
    plrs._expr_nodes.Literal,
    plrs._expr_nodes.Sort,
    plrs._expr_nodes.SortBy,
    plrs._expr_nodes.Gather,
    plrs._expr_nodes.Filter,
    plrs._expr_nodes.Cast,
    plrs._expr_nodes.Column,
    plrs._expr_nodes.Agg,
    plrs._expr_nodes.BinaryExpr,
    plrs._expr_nodes.Len,
    plrs._expr_nodes.PyExprIR,
]

PolarsSchema: TypeAlias = dict[str, pl.DataType]
Schema: TypeAlias = dict[str, "DataType"]

PolarsDataType: TypeAlias = polars.datatypes.DataTypeClass | polars.datatypes.DataType

Slice: TypeAlias = tuple[int, int | None]

CSECache: TypeAlias = MutableMapping[int, tuple["DataFrame", int]]

ClosedInterval: TypeAlias = Literal["left", "right", "both", "none"]

Duration = NewType("Duration", tuple[int, int, int, int, bool, bool])


class NodeTraverser(Protocol):
    """Abstract protocol for polars NodeTraverser."""

    def get_node(self) -> int:
        """Return current plan node id."""
        ...

    def set_node(self, n: int) -> None:
        """Set the current plan node to n."""
        ...

    def view_current_node(self) -> PolarsIR:
        """Convert current plan node to python rep."""
        ...

    def get_schema(self) -> PolarsSchema:
        """Get the schema of the current plan node."""
        ...

    def get_dtype(self, n: int) -> pl.DataType:
        """Get the datatype of the given expression id."""
        ...

    def view_expression(self, n: int) -> PolarsExpr:
        """Convert the given expression to python rep."""
        ...

    def version(self) -> tuple[int, int]:
        """The IR version as `(major, minor)`."""
        ...

    def set_udf(
        self,
        callback: Callable[[list[str] | None, str | None, int | None], pl.DataFrame],
    ) -> None:
        """Set the callback replacing the current node in the plan."""
        ...


OptimizationArgs: TypeAlias = Literal[
    "type_coercion",
    "predicate_pushdown",
    "projection_pushdown",
    "simplify_expression",
    "slice_pushdown",
    "comm_subplan_elim",
    "comm_subexpr_elim",
    "cluster_with_columns",
    "no_optimization",
    "optimizations",
]

CollectKwargs: TypeAlias = dict[OptimizationArgs, bool | pl.QueryOptFlags]


U_contra = TypeVar("U_contra", bound=Hashable, contravariant=True)
V_co = TypeVar("V_co", covariant=True)
StateT_co = TypeVar("StateT_co", covariant=True)
NodeT = TypeVar("NodeT", bound="nodebase.Node[Any]")
ItemT_co = TypeVar("ItemT_co", covariant=True)


class SizedIterator(Iterator[ItemT_co], Sized, Protocol[ItemT_co]):
    """An iterator that also reports how many items it will yield."""


class GenericTransformer(Protocol[U_contra, V_co, StateT_co]):
    """Abstract protocol for recursive visitors."""

    def __call__(self, __value: U_contra) -> V_co:
        """Apply the visitor to the node."""
        ...

    @property
    def state(self) -> StateT_co:
        """Transform-specific immutable state."""
        ...


class _ScalarDataTypeHeader(TypedDict):
    kind: Literal["scalar"]
    name: str


class _DecimalDataTypeHeader(TypedDict):
    kind: Literal["decimal"]
    precision: int
    scale: int


class _DatetimeDataTypeHeader(TypedDict):
    kind: Literal["datetime"]
    time_unit: str
    time_zone: str | None


class _DurationDataTypeHeader(TypedDict):
    kind: Literal["duration"]
    time_unit: str


class _ListDataTypeHeader(TypedDict):
    kind: Literal["list"]
    inner: DataTypeHeader


class _StructFieldHeader(TypedDict):
    name: str
    dtype: DataTypeHeader


class _StructDataTypeHeader(TypedDict):
    kind: Literal["struct"]
    fields: list[_StructFieldHeader]


DataTypeHeader = (
    _ScalarDataTypeHeader
    | _DecimalDataTypeHeader
    | _DatetimeDataTypeHeader
    | _DurationDataTypeHeader
    | _ListDataTypeHeader
    | _StructDataTypeHeader
)


class ColumnOptions(TypedDict):
    """
    Column constructor options.

    Notes
    -----
    Used to serialize Column and DataFrame containers.
    """

    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    name: str | None
    dtype: DataTypeHeader


class DeserializedColumnOptions(TypedDict):
    """
    Deserialized Column constructor options.

    Notes
    -----
    Used to deserialize Column and DataFrame containers.
    """

    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    name: str | None
    dtype: DataType


class ColumnHeader(TypedDict):
    """Column serialization header."""

    column_kwargs: ColumnOptions
    frame_count: int


class DataFrameHeader(TypedDict):
    """DataFrame serialization header."""

    columns_kwargs: list[ColumnOptions]
    frame_count: int


# Not public in polars yet
RankMethod = Literal["ordinal", "dense", "min", "max", "average"]

RoundMethod = Literal["half_away_from_zero", "half_to_even"]
