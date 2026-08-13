# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
DSL nodes for the LogicalPlan of polars.

An IR node is either a source, normal, or a sink. Respectively they
can be considered as functions:

- source: `IO () -> DataFrame`
- normal: `DataFrame -> DataFrame`
- sink: `DataFrame -> IO ()`
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import itertools
import json
import random
import reprlib
import time
import uuid
from collections.abc import Sized
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ParamSpec,
    TypeVar,
    assert_never,
    overload,
)

import polars as pl

import pylibcudf as plc
from pylibcudf import expressions as plc_expr

import cudf_polars.dsl.expr as expr
from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.containers.dataframe import NamedColumn
from cudf_polars.dsl.expressions import rolling, unary
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.nodebase import Node
from cudf_polars.dsl.to_ast import _DECIMAL_IDS, to_ast, to_parquet_filter
from cudf_polars.dsl.tracing import log_do_evaluate, nvtx_annotate_cudf_polars
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.dsl.utils.windows import (
    offsets_to_windows,
    range_window_bounds,
)
from cudf_polars.utils import dtypes
from cudf_polars.utils.cuda_stream import (
    get_cuda_stream,
    stream_ordered_after,
)
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_136,
    POLARS_VERSION_LT_137,
    POLARS_VERSION_LT_138,
)

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import (
        Callable,
        Generator,
        Hashable,
        Iterable,
        Iterator,
        Sequence,
    )
    from typing import Literal, Self

    from polars import polars  # type: ignore[attr-defined]

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers.dataframe import NamedColumn
    from cudf_polars.dsl.utils.io import CachedParquetInfo
    from cudf_polars.streaming.rank_aware_source import RankAwareSource
    from cudf_polars.typing import CSECache, ClosedInterval, Schema, Slice as Zlice
    from cudf_polars.utils.config import ParquetOptions
    from cudf_polars.utils.timer import Timer

    P = ParamSpec("P")
    T = TypeVar("T")

__all__ = [
    "IR",
    "Cache",
    "ConditionalJoin",
    "DataFrameScan",
    "Distinct",
    "Empty",
    "ErrorNode",
    "Filter",
    "GroupBy",
    "HConcat",
    "HStack",
    "IRExecutionContext",
    "Join",
    "MapFunction",
    "MergeSorted",
    "Projection",
    "PythonScan",
    "Reduce",
    "Rolling",
    "Scan",
    "Select",
    "Sink",
    "Slice",
    "Sort",
    "Union",
]


@dataclass(frozen=True)
class IRExecutionContext:
    """
    Runtime context for IR node execution.

    This dataclass holds runtime information and configuration needed
    during the evaluation of IR nodes.

    Parameters
    ----------
    py_executor
       Thread pool for thread offload in async execution, only used by
       streaming engine.
    get_cuda_stream
        A zero-argument callable that returns a CUDA stream.
    query_id
        Identifier for the query being executed.
    """

    py_executor: concurrent.futures.ThreadPoolExecutor | None = field(default=None)
    get_cuda_stream: Callable[[], Stream] = field(default=get_cuda_stream)
    query_id: uuid.UUID = field(default_factory=uuid.uuid4)

    async def to_thread(
        self, func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """
        Run a function asynchronously in a thread.

        Parameters
        ----------
        func
            The function to run.
        args
            Arguments.
        kwargs
            Keyword arguments.

        Returns
        -------
        Awaitable to obtain the result of calling ``func``.

        Notes
        -----
        This offloads the function to run in the thread pool attached to
        this execution context.
        """
        assert self.py_executor is not None, (
            "Execution context must have a thread pool for offload"
        )
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        func_call = functools.partial(ctx.run, func, *args, **kwargs)
        return await loop.run_in_executor(self.py_executor, func_call)

    @contextlib.contextmanager
    def stream_ordered_after(self, *dfs: DataFrame) -> Generator[Stream, None, None]:
        """
        Get a joined CUDA stream with safe stream ordering for deallocation of inputs.

        Parameters
        ----------
        dfs
            The dataframes being provided to stream-ordered operations.

        Yields
        ------
        A CUDA stream that is downstream of the given dataframes.
        """
        with stream_ordered_after(
            self.get_cuda_stream, upstreams=[df.stream for df in dfs]
        ) as result_stream:
            yield result_stream


def apply_predicate(df: DataFrame, predicate: expr.NamedExpr | None) -> DataFrame:
    """Filter ``df`` by a predicate expression."""
    if predicate is None:
        return df
    (mask,) = broadcast(
        predicate.evaluate(df), target_length=df.num_rows, stream=df.stream
    )
    return df.filter(mask)


def concatenate_chunks(
    dfs: Sequence[DataFrame], *, context: IRExecutionContext
) -> DataFrame:
    """Vertically concatenate ``dfs`` (which must share a schema) into one frame."""
    with context.stream_ordered_after(*dfs) as stream:
        return DataFrame.from_table(
            plc.concatenate.concatenate([df.table for df in dfs], stream=stream),
            dfs[0].column_names,
            dfs[0].dtypes,
            stream=stream,
        )


_BINOPS = {
    plc.binaryop.BinaryOperator.EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL,
    # TODO: Handle other binary operations as needed
}


class IR(Node["IR"]):
    """Abstract plan node, representing an unevaluated dataframe."""

    __slots__ = ("_non_child_args", "schema")
    # This annotation is needed because of https://github.com/python/mypy/issues/17981
    _non_child: ClassVar[tuple[str, ...]] = ("schema",)
    # Concrete classes should set this up with the arguments that will
    # be passed to do_evaluate.
    _non_child_args: tuple[Any, ...]
    # The number of non-child arguments to pass to do_evaluate.
    _n_non_child_args: ClassVar[int]
    schema: Schema
    """Mapping from column names to their data types."""

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of node, treating schema dictionary.

        Since the schema is a dictionary, even though it is morally
        immutable, it is not hashable. We therefore convert it to
        tuples for hashing purposes.
        """
        # Schema is the first constructor argument
        args = self._ctor_arguments(self.children)[1:]
        schema_hash = tuple(self.schema.items())
        return (type(self), schema_hash, args)

    # Hacky to avoid type-checking issues, just advertise the
    # signature. Both mypy and pyright complain if we have an abstract
    # method that takes arbitrary *args, but the subclasses have
    # tighter signatures. This complaint is correct because the
    # subclass is not Liskov-substitutable for the superclass.
    # However, we know do_evaluate will only be called with the
    # correct arguments by "construction".
    do_evaluate: Callable[..., DataFrame]
    """
    Evaluate the node (given its evaluated children), and return a dataframe.

    Parameters
    ----------
    args
        Non child arguments followed by any evaluated dataframe inputs.

    Returns
    -------
    DataFrame (on device) representing the evaluation of this plan
    node.

    Raises
    ------
    NotImplementedError
        If evaluation fails. Ideally this should not occur, since the
        translation phase should fail earlier.
    """

    def evaluate(
        self, *, cache: CSECache, timer: Timer | None, context: IRExecutionContext
    ) -> DataFrame:
        """
        Evaluate the node (recursively) and return a dataframe.

        Parameters
        ----------
        cache
            Mapping from cached node ids to constructed DataFrames.
            Used to implement evaluation of the `Cache` node.
        timer
            If not None, a Timer object to record timings for the
            evaluation of the node.
        context
            The execution context for the node.

        Notes
        -----
        Prefer not to override this method. Instead implement
        :meth:`do_evaluate` which doesn't encode a recursion scheme
        and just assumes already evaluated inputs.

        Returns
        -------
        DataFrame (on device) representing the evaluation of this plan
        node (and its children).

        Raises
        ------
        NotImplementedError
            If evaluation fails. Ideally this should not occur, since the
            translation phase should fail earlier.
        """
        children = [
            child.evaluate(cache=cache, timer=timer, context=context)
            for child in self.children
        ]
        if timer is not None:
            start = time.monotonic_ns()
            result = self.do_evaluate(*self._non_child_args, *children, context=context)
            end = time.monotonic_ns()
            # TODO: Set better names on each class object.
            timer.store(start, end, type(self).__name__)
            return result
        else:
            return self.do_evaluate(*self._non_child_args, *children, context=context)


class ErrorNode(IR):
    """Represents an error translating the IR."""

    __slots__ = ("error",)
    _non_child = (
        "schema",
        "error",
    )
    _n_non_child_args = 0
    error: str
    """The error."""

    def __init__(self, schema: Schema, error: str):
        self.schema = schema
        self.error = error
        self.children = ()


class PythonScan(IR):
    """Representation of input from a python function."""

    __slots__ = ("options", "predicate")
    _non_child = ("schema", "options", "predicate")
    _n_non_child_args = 3
    options: Any
    """Arbitrary options."""
    predicate: expr.NamedExpr | None
    """Filter to apply to the constructed dataframe before returning it."""

    def __init__(self, schema: Schema, options: Any, predicate: expr.NamedExpr | None):
        self.schema = schema
        self.options = options
        self.predicate = predicate
        self._non_child_args = (schema, options, predicate)
        self.children = ()

    def get_hashable(self) -> Hashable:
        """Return a hashable representation."""
        _scan_fn, with_columns, source_type, ir_node_index = self.options
        return (
            type(self),
            tuple(self.schema.items()),
            ir_node_index,
            tuple(with_columns) if with_columns is not None else None,
            source_type,
            self.predicate,
        )

    @staticmethod
    def run_source_function(
        options: Any,
        schema: Schema,
        *,
        rank_aware_source: RankAwareSource | None = None,
        rank: int = 0,
        nranks: int = 1,
        context: IRExecutionContext,
    ) -> tuple[int | None, Iterator[pl.DataFrame | DataFrame]]:
        """
        Call the source and return its raw chunks.

        The source's scan function is invoked here, so this is where the
        registered IO source's user code runs. A source that yields no chunks
        produces a single empty host frame carrying the declared schema, so
        the iterator always yields at least one chunk.

        Parameters
        ----------
        options
            The `PythonScan` options tuple
            ``(scan_fn, with_columns, source_type, ir_node_index)``.
        schema
            The declared output schema, used for the empty-source fallback frame.
        rank_aware_source
            The `RankAwareSource` captured by the registered scan function,
            when one was found.
        rank
            Rank running this scan, passed to a `RankAwareSource`. Defaults to
            ``0`` for single-rank / in-memory execution.
        nranks
            Total number of ranks, passed to a `RankAwareSource`. Defaults to
            ``1`` for single-rank / in-memory execution.
        context
            The IR execution context (provides the CUDA stream).

        Returns
        -------
        count
            The number of chunks the source will yield when it reports one up front (a
            sized iterator) or None for a plain generator whose length is unknown until
            it is drained.
        chunks
            An iterator of raw chunks, always yielding at least one chunk.

        Raises
        ------
        TypeError
            If the source yields something other than a polars or cudf-polars
            DataFrame.
        """
        scan_fn, with_columns, _source_type, _ir_node_index = options
        # We pass predicate=None and apply any pushed predicate on the
        # GPU in process_chunk.
        # TODO: forward the pushed predicate to a RankAwareSource so a GPU-aware source
        # can apply it at read time. See https://github.com/rapidsai/cudf/issues/22917.
        if rank_aware_source is not None:
            source_chunks = rank_aware_source(
                with_columns, None, None, None, rank=rank, nranks=nranks
            )
        else:
            # A plain source. The discarded second return value is the wrapper's
            # "predicate applied" flag, which carries no information here.
            source_chunks, _ = scan_fn(with_columns, None, None, None)
        count = len(source_chunks) if isinstance(source_chunks, Sized) else None

        def chunks() -> Generator[pl.DataFrame | DataFrame, None, None]:
            iterator = iter(source_chunks)
            sentinel = object()
            first = next(iterator, sentinel)
            if first is sentinel:
                # Empty source: emit one empty host frame carrying the declared schema.
                yield pl.DataFrame(
                    schema={name: dtype.polars_type for name, dtype in schema.items()}
                )
                return
            for chunk in itertools.chain((first,), iterator):
                if not isinstance(chunk, (DataFrame, pl.DataFrame)):
                    raise TypeError(
                        "PythonScan source must yield polars.DataFrame or "
                        "cudf_polars.containers.DataFrame chunks"
                    )
                yield chunk

        return count, chunks()

    @staticmethod
    def process_chunk(
        chunk: pl.DataFrame | DataFrame,
        schema: Schema,
        predicate: expr.NamedExpr | None,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """
        Move a raw source chunk to the device, validate it, and filter it.

        Parameters
        ----------
        chunk
            A raw chunk returned from :meth:`run_source_function`.
        schema
            The declared output schema. The chunk must match it in name, order,
            and dtype.
        predicate
            A pushed-down filter to apply on the device, or ``None``.
        context
            The IR execution context (provides the CUDA stream).

        Returns
        -------
        The device dataframe for this chunk, filtered by ``predicate``.

        Raises
        ------
        polars.exceptions.SchemaError
            If the chunk's schema does not match ``schema``.
        """
        # A host frame is copied to the device; a cudf-polars frame is already there.
        df = (
            chunk
            if isinstance(chunk, DataFrame)
            else DataFrame.from_polars(chunk, stream=context.get_cuda_stream())
        )
        # Validate against the declared (output) schema. Polars performs this
        # check for register_io_source(..., validate_schema=True), but the flag is
        # not exposed to the GPU plan, so we always validate.
        # See https://github.com/rapidsai/cudf/issues/23043
        declared = pl.Schema(
            {name: dtype.polars_type for name, dtype in schema.items()}
        )
        produced = pl.Schema(
            zip(
                df.column_names,
                (dtype.polars_type for dtype in df.dtypes),
                strict=True,
            )
        )
        if produced != declared:
            raise pl.exceptions.SchemaError(
                f"user provided schema: {declared} doesn't match the "
                f"DataFrame schema: {produced}"
            )
        return apply_predicate(df, predicate)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="PythonScan")
    def do_evaluate(
        cls,
        schema: Schema,
        options: Any,
        predicate: expr.NamedExpr | None,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """
        Evaluate and return a dataframe.

        This runs the source eagerly, producing the whole scan output as one
        frame. It is used by the non-streaming (in-memory) engine. Streaming
        engines do not call this; they handle ``PythonScan`` with a dedicated
        actor (``python_scan_node``) that streams chunks individually.
        """
        _, raw = cls.run_source_function(options, schema, context=context)
        chunks = [
            cls.process_chunk(chunk, schema, predicate, context=context)
            for chunk in raw
        ]
        # run_source_function always yields at least one chunk.
        if len(chunks) == 1:
            return chunks[0]
        return concatenate_chunks(chunks, context=context)


_COMPARISON_BINOPS = {
    plc.binaryop.BinaryOperator.EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL,
}


@nvtx_annotate_cudf_polars(message="_parquet_physical_types")
def _parquet_physical_types(
    paths: list[str], columns: list[str] | None
) -> dict[str, plc.DataType]:
    # TODO: Use prefetched metadata
    # https://github.com/rapidsai/cudf/issues/22940
    metadata = plc.io.parquet_metadata.read_parquet_metadata(plc.io.SourceInfo(paths))
    column_types = metadata.schema().column_types()

    if columns is not None:
        return {name: column_types[name] for name in columns if name in column_types}
    return column_types  # pragma: no cover


def _cast_literal_to_decimal(
    side: expr.Expr, lit: expr.Literal, phys_type_map: dict[str, plc.DataType]
) -> expr.Expr:
    if isinstance(side, expr.Cast):
        col = side.children[0]
        assert isinstance(col, expr.Col)
        name = col.name
    else:
        assert isinstance(side, expr.Col)
        name = side.name
    if (type_ := phys_type_map[name]).id() in _DECIMAL_IDS:
        scale = abs(type_.scale())
        return expr.Cast(
            side.dtype,
            True,  # noqa: FBT003
            expr.Cast(DataType(pl.Decimal(38, scale)), True, lit),  # noqa: FBT003
        )
    return lit


def _cast_literals_to_physical_types(
    node: expr.Expr, phys_type_map: dict[str, plc.DataType]
) -> expr.Expr:
    if isinstance(node, expr.BinOp):
        left, right = node.children
        left = _cast_literals_to_physical_types(left, phys_type_map)
        right = _cast_literals_to_physical_types(right, phys_type_map)
        if node.op in _COMPARISON_BINOPS:
            if isinstance(left, (expr.Col, expr.Cast)) and isinstance(
                right, expr.Literal
            ):
                right = _cast_literal_to_decimal(left, right, phys_type_map)
            elif isinstance(right, (expr.Col, expr.Cast)) and isinstance(
                left, expr.Literal
            ):
                left = _cast_literal_to_decimal(right, left, phys_type_map)

        return node.reconstruct([left, right])
    return node


def _prepare_parquet_predicate(
    predicate: expr.Expr,
    paths: list[str],
    schema: Schema,
    columns: list[str] | None,
) -> expr.Expr:
    cols = columns or list(schema.keys())
    if any(isinstance(schema[c].polars_type, pl.Decimal) for c in cols if c in schema):
        return _cast_literals_to_physical_types(
            predicate, _parquet_physical_types(paths, cols)
        )
    return predicate


class Scan(IR):
    """Input from files."""

    __slots__ = (
        "cached_parquet_info",
        "cloud_options",
        "include_file_paths",
        "n_rows",
        "parquet_options",
        "paths",
        "predicate",
        "reader_options",
        "row_index",
        "skip_rows",
        "typ",
        "with_columns",
    )
    _non_child = (
        "schema",
        "typ",
        "reader_options",
        "cloud_options",
        "paths",
        "with_columns",
        "skip_rows",
        "n_rows",
        "row_index",
        "include_file_paths",
        "predicate",
        "parquet_options",
    )
    _n_non_child_args = 12
    typ: str
    """What type of file are we reading? Parquet, CSV, etc..."""
    reader_options: dict[str, Any]
    """Reader-specific options, as dictionary."""
    cloud_options: dict[str, Any] | None
    """Cloud-related authentication options, currently ignored."""
    paths: list[str]
    """List of paths to read from."""
    with_columns: list[str] | None
    """Projected columns to return."""
    skip_rows: int
    """Rows to skip at the start when reading."""
    n_rows: int
    """Number of rows to read after skipping."""
    row_index: tuple[str, int] | None
    """If not None add an integer index column of the given name."""
    include_file_paths: str | None
    """Include the path of the source file(s) as a column with this name."""
    predicate: expr.NamedExpr | None
    """Mask to apply to the read dataframe."""
    parquet_options: ParquetOptions
    """Parquet-specific options."""
    cached_parquet_info: list[CachedParquetInfo] | None
    """Cached parquet file metadata."""

    PARQUET_DEFAULT_CHUNK_SIZE: int = 0  # unlimited
    PARQUET_DEFAULT_PASS_LIMIT: int = 16 * 1024**3  # 16GiB

    def __init__(
        self,
        schema: Schema,
        typ: str,
        reader_options: dict[str, Any],
        cloud_options: dict[str, Any] | None,
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
        predicate: expr.NamedExpr | None,
        parquet_options: ParquetOptions,
        cached_parquet_info: list[CachedParquetInfo] | None = None,
    ):
        self.schema = schema
        self.typ = typ
        self.reader_options = reader_options
        self.cloud_options = cloud_options
        self.paths = paths
        self.with_columns = with_columns
        self.skip_rows = skip_rows
        self.n_rows = n_rows
        self.row_index = row_index
        self.include_file_paths = include_file_paths
        self.predicate = predicate
        self._non_child_args = (
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
        )
        self.children = ()
        self.parquet_options = parquet_options
        self.cached_parquet_info = cached_parquet_info

        Scan._validate_cached_parquet_info(self.paths, self.cached_parquet_info)

        if self.typ not in ("csv", "parquet", "ndjson"):  # pragma: no cover
            # This line is unhittable ATM since IPC/Anonymous scan raise
            # on the polars side
            raise NotImplementedError(f"Unhandled scan type: {self.typ}")
        if self.typ == "ndjson" and (self.n_rows != -1 or self.skip_rows != 0):
            raise NotImplementedError("row limit in scan for json reader")
        if self.skip_rows < 0:
            # TODO: polars has this implemented for parquet,
            # maybe we can do this too?
            raise NotImplementedError("slice pushdown for negative slices")
        if self.cloud_options is not None and any(
            self.cloud_options.get(k) is not None for k in ("aws", "azure", "gcp")
        ):
            raise NotImplementedError(
                "Read from cloud storage"
            )  # pragma: no cover; no test yet
        if self.typ == "csv":
            if any(
                plc.io.SourceInfo._is_remote_uri(p) for p in self.paths
            ):  # pragma: no cover; no test yet
                # This works fine when the file has no leading blank lines,
                # but currently we do some file introspection
                # to skip blanks before parsing the header.
                # For remote files we cannot determine if leading blank lines
                # exist, so we're punting on CSV support.
                # TODO: Once the CSV reader supports skipping leading
                # blank lines natively, we can remove this guard.
                raise NotImplementedError(
                    "Reading CSV from remote is not yet supported"
                )

            if self.reader_options["skip_rows_after_header"] != 0:
                raise NotImplementedError("Skipping rows after header in CSV reader")
            parse_options = self.reader_options["parse_options"]
            if (
                null_values := parse_options["null_values"]
            ) is not None and "Named" in null_values:
                raise NotImplementedError(
                    "Per column null value specification not supported for CSV reader"
                )
            if (
                comment := parse_options["comment_prefix"]
            ) is not None and "Multi" in comment:
                raise NotImplementedError(
                    "Multi-character comment prefix not supported for CSV reader"
                )
            if not self.reader_options["has_header"]:
                # TODO: To support reading headerless CSV files without requiring new
                # column names, we would need to do file introspection to infer the number
                # of columns so column projection works right.
                reader_schema = self.reader_options.get("schema")
                if not (
                    reader_schema
                    and isinstance(schema, dict)
                    and "fields" in reader_schema
                ):
                    raise NotImplementedError(
                        "Reading CSV without header requires user-provided column names via new_columns"
                    )
        elif self.typ == "ndjson":
            # TODO: consider handling the low memory option here
            # (maybe use chunked JSON reader)
            if self.reader_options["ignore_errors"]:
                raise NotImplementedError(
                    "ignore_errors is not supported in the JSON reader"
                )
            if include_file_paths is not None:
                # TODO: Need to populate num_rows_per_source in read_json in libcudf
                raise NotImplementedError("Including file paths in a json scan.")
        elif (
            self.typ == "parquet"
            and self.row_index is not None
            and self.with_columns is not None
            and len(self.with_columns) == 0
        ):
            raise NotImplementedError(
                "Reading only parquet metadata to produce row index."
            )

    @staticmethod
    def _validate_cached_parquet_info(
        paths: list[str],
        cached_parquet_info: list[CachedParquetInfo] | None,
    ) -> None:
        if cached_parquet_info is not None and paths != [
            info.path for info in cached_parquet_info
        ]:
            missing = reprlib.repr(
                set(paths) - {info.path for info in cached_parquet_info}
            )
            raise AssertionError(
                f"Paths do not match cached parquet info. Missing paths: {missing}"
            )

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of the node.

        The options dictionaries are serialised for hashing purposes
        as json strings.
        """
        schema_hash = tuple(self.schema.items())
        return (
            type(self),
            schema_hash,
            self.typ,
            json.dumps(self.reader_options),
            json.dumps(self.cloud_options),
            tuple(self.paths),
            tuple(self.with_columns) if self.with_columns is not None else None,
            self.skip_rows,
            self.n_rows,
            self.row_index,
            self.include_file_paths,
            self.predicate,
            self.parquet_options,
        )

    @staticmethod
    def add_file_paths(
        name: str, paths: list[str], rows_per_path: list[int], df: DataFrame
    ) -> DataFrame:
        """
        Add a Column of file paths to the DataFrame.

        Each path is repeated according to the number of rows read from it.
        """
        (filepaths,) = plc.filling.repeat(
            plc.Table(
                [
                    plc.Column.from_arrow(
                        pl.Series(values=map(str, paths)),
                        stream=df.stream,
                    )
                ]
            ),
            plc.Column.from_arrow(
                pl.Series(values=rows_per_path, dtype=pl.datatypes.Int32()),
                stream=df.stream,
            ),
            stream=df.stream,
        ).columns()
        dtype = DataType(pl.String())
        return df.with_columns(
            [Column(filepaths, name=name, dtype=dtype)], stream=df.stream
        )

    @staticmethod
    @nvtx_annotate_cudf_polars(message="Scan._get_parquet_row_count_from_metadata")
    def _get_parquet_row_count_from_metadata(
        paths: list[str],
        skip_rows: int,
        n_rows: int,
        parquet_options: ParquetOptions,
        cached_parquet_info: list[CachedParquetInfo] | None,
    ) -> int:
        # Zero-width parquet files lose their row count when read through
        # pylibcudf. See https://github.com/rapidsai/cudf/issues/21428
        if parquet_options.prefetch_file_metadata:
            if cached_parquet_info is None:
                raise AssertionError(
                    "Cached parquet info is required when prefetching file metadata is enabled"
                )

            Scan._validate_cached_parquet_info(paths, cached_parquet_info)
            parquet_metadatas = [
                info.file_metadata for info in cached_parquet_info
            ]  # pragma: no cover
            num_rows = sum(
                metadata.num_rows for metadata in parquet_metadatas
            )  # pragma: no cover
        else:
            meta = plc.io.parquet_metadata.read_parquet_metadata(
                plc.io.SourceInfo(paths)
            )
            num_rows = meta.num_rows()

        num_rows -= skip_rows
        if n_rows != -1:
            num_rows = min(num_rows, n_rows)
        return max(num_rows, 0)

    @staticmethod
    def _apply_parquet_projection(
        table: plc.Table, names: Sequence[str], with_columns: Sequence[str] | None
    ) -> tuple[plc.Table, list[str]]:
        # `set_column_names` may leave pandas-indices in the table.
        # Here we ensure that the table only contains the projection.
        if with_columns is None:
            return table, list(names)
        names = list(names)
        with_columns = list(with_columns)
        columns = dict(zip(names, table.columns(), strict=True))
        return plc.Table([columns[name] for name in with_columns]), with_columns

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Scan")
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
        predicate: expr.NamedExpr | None,
        parquet_options: ParquetOptions,
        cached_parquet_info: list[CachedParquetInfo] | None,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        stream = context.get_cuda_stream()
        if typ == "csv":

            def read_csv_header(
                path: Path | str, sep: str
            ) -> list[str]:  # pragma: no cover
                with Path(path).open(encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            return stripped.split(sep)
                return []

            parse_options = reader_options["parse_options"]
            sep = chr(parse_options["separator"])
            quote = chr(parse_options["quote_char"])
            eol = chr(parse_options["eol_char"])
            if reader_options["schema"] is not None:
                # Reader schema provides names
                column_names = list(reader_options["schema"]["fields"].keys())
            else:
                # file provides column names
                column_names = None
            usecols = with_columns
            has_header = reader_options["has_header"]
            header = 0 if has_header else -1

            # polars defaults to no null recognition
            null_values = [""]
            if parse_options["null_values"] is not None:
                ((typ, nulls),) = parse_options["null_values"].items()
                if typ == "AllColumnsSingle":
                    # Single value
                    null_values.append(nulls)
                else:
                    # List of values
                    null_values.extend(nulls)
            if parse_options["comment_prefix"] is not None:
                comment = chr(parse_options["comment_prefix"]["Single"])
            else:
                comment = None
            decimal = "," if parse_options["decimal_comma"] else "."

            # polars skips blank lines at the beginning of the file
            pieces = []
            seen_paths = []
            read_partial = n_rows != -1
            for p in paths:
                skiprows = reader_options["skip_rows"]
                path = Path(p)
                with path.open(encoding="utf-8") as f:
                    while f.readline() == "\n":
                        skiprows += 1
                options = (
                    plc.io.csv.CsvReaderOptions.builder(plc.io.SourceInfo([path]))
                    .nrows(n_rows)
                    .skiprows(skiprows + skip_rows)
                    .skip_blank_lines(skip_blank_lines=False)
                    .lineterminator(str(eol))
                    .quotechar(str(quote))
                    .decimal(decimal)
                    .keep_default_na(keep_default_na=False)
                    .na_filter(na_filter=True)
                    .delimiter(str(sep))
                    .build()
                )
                if column_names is not None:
                    options.set_names([str(name) for name in column_names])
                else:
                    if header > -1 and skip_rows > header:  # pragma: no cover
                        # We need to read the header otherwise we would skip it
                        column_names = read_csv_header(path, str(sep))
                        options.set_names(column_names)
                options.set_header(header)
                options.set_dtypes(
                    {name: dtype.plc_type for name, dtype in schema.items()}
                )
                if usecols is not None:
                    options.set_use_cols_names([str(name) for name in usecols])
                options.set_na_values(null_values)
                if comment is not None:
                    options.set_comment(comment)
                tbl_w_meta = plc.io.csv.read_csv(options, stream=stream)
                pieces.append(tbl_w_meta)
                if include_file_paths is not None:
                    seen_paths.append(p)
                if read_partial:
                    n_rows -= tbl_w_meta.tbl.num_rows()
                    if n_rows <= 0:
                        break
            tables, (colnames, *_) = zip(
                *(
                    (piece.tbl, piece.column_names(include_children=False))
                    for piece in pieces
                ),
                strict=True,
            )
            df = DataFrame.from_table(
                plc.concatenate.concatenate(list(tables), stream=stream),
                colnames,
                [schema[colname] for colname in colnames],
                stream=stream,
            )
            if include_file_paths is not None:
                df = Scan.add_file_paths(
                    include_file_paths,
                    seen_paths,
                    [t.num_rows() for t in tables],
                    df,
                )
        elif typ == "parquet":
            if parquet_options.prefetch_file_metadata:
                if cached_parquet_info is None:
                    raise AssertionError(
                        "Cached parquet info is required when prefetching file metadata is enabled"
                    )
                Scan._validate_cached_parquet_info(paths, cached_parquet_info)
                filepath_sources = []
                parquet_metadatas = []
                for info in cached_parquet_info:
                    filepath_sources.append(
                        plc.io.types.FilepathSource(info.path, info.size)
                    )
                    parquet_metadatas.append(info.file_metadata)
                source_info = plc.io.SourceInfo(filepath_sources)
            else:
                parquet_metadatas = None
                source_info = plc.io.SourceInfo(paths)

            filters = None
            if predicate is not None and row_index is None:
                # Can't apply filters during read if we have a row index.
                filters = to_parquet_filter(
                    _prepare_parquet_predicate(
                        predicate.value, paths, schema, with_columns
                    ),
                    stream=stream,
                )
            builder = plc.io.parquet.ParquetReaderOptions.builder(source_info)
            if filters is not None and parquet_options.use_jit_filter:
                builder.use_jit_filter(use_jit_filter=True)
            parquet_reader_options = builder.decimal_width(
                plc.TypeId.DECIMAL128
            ).build()

            if with_columns is not None:
                parquet_reader_options.set_column_names(with_columns)
            if filters is not None:
                parquet_reader_options.set_filter(filters)
            if n_rows != -1:
                parquet_reader_options.set_num_rows(n_rows)
            if skip_rows != 0:
                parquet_reader_options.set_skip_rows(skip_rows)
            if parquet_options.chunked:
                reader = plc.io.parquet.ChunkedParquetReader(
                    parquet_reader_options,
                    chunk_read_limit=parquet_options.chunk_read_limit,
                    pass_read_limit=parquet_options.pass_read_limit,
                    parquet_metadatas=parquet_metadatas,
                    stream=stream,
                )
                chunk = reader.read_chunk()
                # TODO: Nested column names
                names = chunk.column_names(include_children=False)
                concatenated_columns = chunk.tbl.release()
                while reader.has_next():
                    columns = reader.read_chunk().tbl.release()
                    # Discard columns while concatenating to reduce memory footprint.
                    # Reverse order to avoid O(n^2) list popping cost.
                    for i in reversed(range(len(concatenated_columns))):
                        concatenated_columns[i] = plc.concatenate.concatenate(
                            [concatenated_columns[i], columns.pop()], stream=stream
                        )
                table, names = cls._apply_parquet_projection(
                    plc.Table(concatenated_columns), names, with_columns
                )
                if not names:
                    table = plc.Table(
                        table.columns(),
                        num_rows=cls._get_parquet_row_count_from_metadata(
                            paths,
                            skip_rows,
                            n_rows,
                            parquet_options,
                            cached_parquet_info,
                        ),
                    )
                df = DataFrame.from_table(
                    table,
                    names=names,
                    dtypes=[schema[name] for name in names],
                    stream=stream,
                )
                if include_file_paths is not None:
                    df = Scan.add_file_paths(  # pragma: no cover
                        include_file_paths, paths, chunk.num_rows_per_source, df
                    )
            else:
                tbl_w_meta = plc.io.parquet.read_parquet(
                    parquet_reader_options,
                    parquet_metadatas=parquet_metadatas,
                    stream=stream,
                )
                # TODO: consider nested column names?
                col_names = tbl_w_meta.column_names(include_children=False)
                table, col_names = cls._apply_parquet_projection(
                    tbl_w_meta.tbl, col_names, with_columns
                )
                if not col_names:
                    table = plc.Table(
                        table.columns(),
                        num_rows=cls._get_parquet_row_count_from_metadata(
                            paths,
                            skip_rows,
                            n_rows,
                            parquet_options,
                            cached_parquet_info,
                        ),
                    )
                df = DataFrame.from_table(
                    table,
                    col_names,
                    [schema[name] for name in col_names],
                    stream=stream,
                )
                if include_file_paths is not None:
                    df = Scan.add_file_paths(
                        include_file_paths, paths, tbl_w_meta.num_rows_per_source, df
                    )
            if filters is not None:
                # Mask must have been applied.
                return df
        elif typ == "ndjson":
            json_schema: list[plc.io.json.NameAndType] = [
                (name, typ.plc_type, []) for name, typ in schema.items()
            ]
            json_reader_options = (
                plc.io.json.JsonReaderOptions.builder(plc.io.SourceInfo(paths))
                .lines(val=True)
                .dtypes(json_schema)
                .prune_columns(val=True)
                .build()
            )
            plc_tbl_w_meta = plc.io.json.read_json(json_reader_options, stream=stream)
            # TODO: I don't think cudf-polars supports nested types in general right now
            # (but when it does, we should pass child column names from nested columns in)
            col_names = plc_tbl_w_meta.column_names(include_children=False)
            df = DataFrame.from_table(
                plc_tbl_w_meta.tbl,
                col_names,
                [schema[name] for name in col_names],
                stream=stream,
            )
            col_order = list(schema.keys())
            if row_index is not None:
                col_order.remove(row_index[0])
            df = df.select(col_order)
        else:
            raise NotImplementedError(
                f"Unhandled scan type: {typ}"
            )  # pragma: no cover; post init trips first
        if row_index is not None:
            name, offset = row_index
            offset += skip_rows
            dtype = schema[name]
            step = plc.Scalar.from_py(1, dtype.plc_type, stream=stream)
            init = plc.Scalar.from_py(offset, dtype.plc_type, stream=stream)
            index_col = Column(
                plc.filling.sequence(df.num_rows, init, step, stream=stream),
                is_sorted=plc.types.Sorted.YES,
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.AFTER,
                name=name,
                dtype=dtype,
            )
            df = DataFrame([index_col, *df.columns], stream=df.stream)
            # Reorder to the row-index column is always first.
            df = df.select(schema) if next(iter(schema)) != name else df
        assert all(
            c.obj.type() == schema[name].plc_type for name, c in df.column_map.items()
        )
        return apply_predicate(df, predicate)


class Sink(IR):
    """Sink a dataframe to a file."""

    __slots__ = ("cloud_options", "kind", "options", "parquet_options", "path")
    _non_child = (
        "schema",
        "kind",
        "path",
        "parquet_options",
        "options",
        "cloud_options",
    )
    _n_non_child_args = 5

    kind: str
    """The type of file to write to. Eg. Parquet, CSV, etc."""
    path: str
    """The path to write to"""
    parquet_options: ParquetOptions
    """GPU-specific configuration options"""
    cloud_options: dict[str, Any] | None
    """Cloud-related authentication options, currently ignored."""
    options: dict[str, Any]
    """Sink options from Polars"""

    def __init__(
        self,
        schema: Schema,
        kind: str,
        path: str,
        parquet_options: ParquetOptions,
        options: dict[str, Any],
        cloud_options: dict[str, Any],
        df: IR,
    ):
        self.schema = schema
        self.kind = kind
        self.path = path
        self.parquet_options = parquet_options
        self.options = options
        self.cloud_options = cloud_options
        self.children = (df,)
        self._non_child_args = (schema, kind, path, parquet_options, options)
        if self.cloud_options is not None and any(
            self.cloud_options.get(k) is not None
            for k in ("config", "credential_provider")
        ):
            raise NotImplementedError(
                "Write to cloud storage"
            )  # pragma: no cover; no test yet
        sync_on_close = options.get("sync_on_close")
        if sync_on_close not in {"None", None}:
            raise NotImplementedError(
                f"sync_on_close='{sync_on_close}' is not supported."
            )  # pragma: no cover; no test yet
        child_schema = df.schema.values()
        if kind == "Csv":
            if not all(
                plc.io.csv.is_supported_write_csv(dtype.plc_type)
                for dtype in child_schema
            ):
                # Nested types are unsupported in polars and libcudf
                raise NotImplementedError(
                    "Contains unsupported types for CSV writing"
                )  # pragma: no cover
            serialize = options["serialize_options"]
            if options["include_bom"]:
                raise NotImplementedError("include_bom is not supported.")
            for key in (
                "date_format",
                "time_format",
                "datetime_format",
                "float_scientific",
                "float_precision",
            ):
                if serialize[key] is not None:
                    raise NotImplementedError(f"{key} is not supported.")
            if serialize["quote_style"] != "Necessary":
                raise NotImplementedError("Only quote_style='Necessary' is supported.")
            if chr(serialize["quote_char"]) != '"':
                raise NotImplementedError("Only quote_char='\"' is supported.")
        elif kind == "Parquet":
            compression = options["compression"]
            if isinstance(compression, dict):
                if len(compression) != 1:
                    raise NotImplementedError(
                        "Compression dict with more than one entry."
                    )  # pragma: no cover
                compression, compression_level = next(iter(compression.items()))
                options["compression"] = compression
                if compression_level is not None:
                    raise NotImplementedError(
                        "Setting compression_level is not supported."
                    )
            if compression == "Lz4Raw":
                compression = "Lz4"
                options["compression"] = compression
            if (
                compression != "Uncompressed"
                and not plc.io.parquet.is_supported_write_parquet(
                    getattr(plc.io.types.CompressionType, compression.upper())
                )
            ):
                raise NotImplementedError(
                    f"Compression type '{compression}' is not supported."
                )
        elif kind == (
            "Json" if POLARS_VERSION_LT_137 else "NDJson"
        ):  # pragma: no cover; options are validated on the polars side
            if not all(
                plc.io.json.is_supported_write_json(dtype.plc_type)
                for dtype in child_schema
            ):
                # Nested types are unsupported in polars and libcudf
                raise NotImplementedError(
                    "Contains unsupported types for JSON writing"
                )  # pragma: no cover
            shared_writer_options = {"sync_on_close", "maintain_order", "mkdir"}
            if set(options) - shared_writer_options:
                raise NotImplementedError("Unsupported options passed JSON writer.")
        else:
            raise NotImplementedError(
                f"Unhandled sink kind: {kind}"
            )  # pragma: no cover

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of the node.

        The option dictionary is serialised for hashing purposes.
        """
        schema_hash = tuple(self.schema.items())  # pragma: no cover
        return (
            type(self),
            schema_hash,
            self.kind,
            self.path,
            self.parquet_options,
            json.dumps(self.options),
            json.dumps(self.cloud_options),
        )  # pragma: no cover

    @classmethod
    def _write_csv(
        cls, target: plc.io.SinkInfo, options: dict[str, Any], df: DataFrame
    ) -> None:
        """Write CSV data to a sink."""
        serialize = options["serialize_options"]
        csv_writer_options = (
            plc.io.csv.CsvWriterOptions.builder(target, df.table)
            .include_header(options["include_header"])
            .names(df.column_names if options["include_header"] else [])
            .na_rep(serialize["null"])
            .line_terminator(serialize["line_terminator"])
            .inter_column_delimiter(chr(serialize["separator"]))
            .build()
        )
        plc.io.csv.write_csv(csv_writer_options, stream=df.stream)

    @classmethod
    def _write_json(cls, target: plc.io.SinkInfo, df: DataFrame) -> None:
        """Write Json data to a sink."""
        metadata = plc.io.TableWithMetadata(
            df.table, [(col, []) for col in df.column_names]
        )
        options = (
            plc.io.json.JsonWriterOptions.builder(target, df.table)
            .lines(val=True)
            .na_rep("null")
            .include_nulls(val=True)
            .metadata(metadata)
            .utf8_escaped(val=False)
            .build()
        )
        plc.io.json.write_json(options, stream=df.stream)

    @staticmethod
    def _make_parquet_metadata(df: DataFrame) -> plc.io.types.TableInputMetadata:
        """Create TableInputMetadata and set column names."""
        metadata = plc.io.types.TableInputMetadata(df.table)
        for i, name in enumerate(df.column_names):
            metadata.column_metadata[i].set_name(name)
        return metadata

    @overload
    @staticmethod
    def _apply_parquet_writer_options(
        builder: plc.io.parquet.ChunkedParquetWriterOptionsBuilder,
        options: dict[str, Any],
    ) -> plc.io.parquet.ChunkedParquetWriterOptionsBuilder: ...

    @overload
    @staticmethod
    def _apply_parquet_writer_options(
        builder: plc.io.parquet.ParquetWriterOptionsBuilder,
        options: dict[str, Any],
    ) -> plc.io.parquet.ParquetWriterOptionsBuilder: ...

    @staticmethod
    def _apply_parquet_writer_options(
        builder: plc.io.parquet.ChunkedParquetWriterOptionsBuilder
        | plc.io.parquet.ParquetWriterOptionsBuilder,
        options: dict[str, Any],
    ) -> (
        plc.io.parquet.ChunkedParquetWriterOptionsBuilder
        | plc.io.parquet.ParquetWriterOptionsBuilder
    ):
        """Apply writer options to the builder."""
        compression = options.get("compression")
        if compression and compression != "Uncompressed":
            compression_type = getattr(
                plc.io.types.CompressionType, compression.upper()
            )
            builder = builder.compression(compression_type)

        if (data_page_size := options.get("data_page_size")) is not None:
            builder = builder.max_page_size_bytes(data_page_size)

        if (row_group_size := options.get("row_group_size")) is not None:
            builder = builder.row_group_size_rows(row_group_size)

        return builder

    @classmethod
    def _write_parquet(
        cls,
        target: plc.io.SinkInfo,
        parquet_options: ParquetOptions,
        options: dict[str, Any],
        df: DataFrame,
    ) -> None:
        metadata: plc.io.types.TableInputMetadata = cls._make_parquet_metadata(df)

        builder: (
            plc.io.parquet.ChunkedParquetWriterOptionsBuilder
            | plc.io.parquet.ParquetWriterOptionsBuilder
        )

        if (  # pragma: no cover
            parquet_options.chunked
            and parquet_options.n_output_chunks != 1
            and df.table.num_rows() != 0
        ):
            chunked_builder = plc.io.parquet.ChunkedParquetWriterOptions.builder(
                target
            ).metadata(metadata)
            chunked_builder = cls._apply_parquet_writer_options(
                chunked_builder, options
            )
            chunked_writer_options = chunked_builder.build()
            writer = plc.io.parquet.ChunkedParquetWriter.from_options(
                chunked_writer_options, stream=df.stream
            )

            # TODO: Can be based on a heuristic that estimates chunk size
            # from the input table size and available GPU memory.
            num_chunks = parquet_options.n_output_chunks
            table_chunks = plc.copying.split(
                df.table,
                [i * df.table.num_rows() // num_chunks for i in range(1, num_chunks)],
                stream=df.stream,
            )
            for chunk in table_chunks:
                writer.write(chunk)
            writer.close([])

        else:
            builder = plc.io.parquet.ParquetWriterOptions.builder(
                target, df.table
            ).metadata(metadata)
            builder = cls._apply_parquet_writer_options(builder, options)
            writer_options = builder.build()
            plc.io.parquet.write_parquet(writer_options, stream=df.stream)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Sink")
    def do_evaluate(
        cls,
        schema: Schema,
        kind: str,
        path: str,
        parquet_options: ParquetOptions,
        options: dict[str, Any],
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Write the dataframe to a file."""
        target = plc.io.SinkInfo([path])

        if POLARS_VERSION_LT_136 and options.get("mkdir", False):
            Path(path).parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
        if kind == "Csv":
            cls._write_csv(target, options, df)
        elif kind == "Parquet":
            cls._write_parquet(target, parquet_options, options, df)
        elif kind == ("Json" if POLARS_VERSION_LT_137 else "NDJson"):
            cls._write_json(target, df)

        return DataFrame([], stream=df.stream)


class Cache(IR):
    """
    Return a cached plan node.

    Used for CSE at the plan level.
    """

    __slots__ = ("key", "refcount")
    _non_child = ("schema", "key", "refcount")
    _n_non_child_args = 2
    key: int
    """The cache key."""
    refcount: int | None
    """The number of cache hits."""

    def __init__(self, schema: Schema, key: int, refcount: int | None, value: IR):
        self.schema = schema
        self.key = key
        self.refcount = refcount
        self.children = (value,)
        self._non_child_args = (key, refcount)

    def get_hashable(self) -> Hashable:  # noqa: D102
        # Polars arranges that the keys are unique across all cache
        # nodes that reference the same child, so we don't need to
        # hash the child.
        return (type(self), self.key, self.refcount)

    def is_equal(self, other: Self) -> bool:  # noqa: D102
        if self.key == other.key and self.refcount == other.refcount:
            self.children = other.children
            return True
        return False

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Cache")
    def do_evaluate(
        cls,
        key: int,
        refcount: int | None,
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:  # pragma: no cover; basic evaluation never calls this
        """Evaluate and return a dataframe."""
        # Our value has already been computed for us, so let's just
        # return it.
        return df

    def evaluate(
        self, *, cache: CSECache, timer: Timer | None, context: IRExecutionContext
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # We must override the recursion scheme because we don't want
        # to recurse if we're in the cache.
        try:
            (result, hits) = cache[self.key]
        except KeyError:
            (value,) = self.children
            result = value.evaluate(cache=cache, timer=timer, context=context)
            cache[self.key] = (result, 0)
            return result
        else:
            if self.refcount is None:
                return result

            hits += 1  # pragma: no cover
            if hits == self.refcount:  # pragma: no cover
                del cache[self.key]
            else:  # pragma: no cover
                cache[self.key] = (result, hits)
            return result  # pragma: no cover


class DataFrameScan(IR):
    """
    Input from an existing polars DataFrame.

    This typically arises from ``q.collect().lazy()``
    """

    __slots__ = ("_id_for_hash", "df", "projection")
    _non_child = ("schema", "df", "projection")
    _n_non_child_args = 3
    df: Any
    """Polars internal PyDataFrame object."""
    projection: tuple[str, ...] | None
    """List of columns to project out."""

    def __init__(
        self,
        schema: Schema,
        df: Any,
        projection: Sequence[str] | None,
    ):
        self.schema = schema
        self.df = df
        self.projection = tuple(projection) if projection is not None else None
        self._non_child_args = (
            schema,
            pl.DataFrame._from_pydf(df),
            self.projection,
        )
        self.children = ()
        self._id_for_hash = random.randint(0, 2**64 - 1)

    @staticmethod
    def _reconstruct(
        schema: Schema,
        pl_df: pl.DataFrame,
        projection: Sequence[str] | None,
        id_for_hash: int,
    ) -> DataFrameScan:  # pragma: no cover
        """
        Reconstruct a DataFrameScan from pickled data.

        Parameters
        ----------
        schema: Schema
            The schema of the DataFrameScan.
        pl_df: pl.DataFrame
            The underlying polars DataFrame.
        projection: Sequence[str] | None
            The projection of the DataFrameScan.
        id_for_hash: int
            The id for hash of the DataFrameScan.

        Returns
        -------
        The reconstructed DataFrameScan.
        """
        node = DataFrameScan(schema, pl_df._df, projection)
        node._id_for_hash = id_for_hash
        return node

    def __reduce__(self) -> tuple[Any, ...]:  # pragma: no cover
        """Pickle a DataFrameScan object."""
        return (
            self._reconstruct,
            (*self._non_child_args, self._id_for_hash),
        )

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of the node.

        The (heavy) dataframe object is not hashed. No two instances of
        ``DataFrameScan`` will have the same hash, even if they have the
        same schema, projection, and config options, and data.
        """
        schema_hash = tuple(self.schema.items())
        return (
            type(self),
            schema_hash,
            self._id_for_hash,
            self.projection,
        )

    def is_equal(self, other: Self) -> bool:
        """Equality of DataFrameScan nodes."""
        return self is other or (
            self._id_for_hash == other._id_for_hash
            and self.schema == other.schema
            and self.projection == other.projection
            and pl.DataFrame._from_pydf(self.df).equals(
                pl.DataFrame._from_pydf(other.df)
            )
        )

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="DataFrameScan")
    def do_evaluate(
        cls,
        schema: Schema,
        df: Any,
        projection: tuple[str, ...] | None,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        pl_df = pl.DataFrame._from_pydf(df) if not isinstance(df, pl.DataFrame) else df
        height = pl_df.height

        if projection is not None:
            df = df.select(projection)

        # Zero-width dataframes lose their row count when converted through
        # pylibcudf. See https://github.com/rapidsai/cudf/issues/21428
        if len(schema) == 0:
            return DataFrame([], stream=context.get_cuda_stream(), num_rows=height)

        df = DataFrame.from_polars(df, stream=context.get_cuda_stream())
        assert all(
            c.obj.type() == dtype.plc_type
            for c, dtype in zip(df.columns, schema.values(), strict=True)
        )
        return df


class Select(IR):
    """Produce a new dataframe selecting given expressions from an input."""

    __slots__ = ("exprs", "should_broadcast")
    _non_child = ("schema", "exprs", "should_broadcast")
    _n_non_child_args = 2
    exprs: tuple[expr.NamedExpr, ...]
    """List of expressions to evaluate to form the new dataframe."""
    should_broadcast: bool
    """Should columns be broadcast?"""

    def __init__(
        self,
        schema: Schema,
        exprs: Sequence[expr.NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        df: IR,
    ):
        self.schema = schema
        self.exprs = tuple(exprs)
        self.should_broadcast = should_broadcast
        self.children = (df,)
        self._non_child_args = (self.exprs, should_broadcast)
        if (
            Select._is_len_expr(self.exprs)
            and isinstance(df, Scan)
            and df.typ != "parquet"
        ):  # pragma: no cover
            raise NotImplementedError(f"Unsupported scan type: {df.typ}")

    @staticmethod
    def _is_len_expr(exprs: tuple[expr.NamedExpr, ...]) -> bool:  # pragma: no cover
        if len(exprs) == 1:
            expr0 = exprs[0].value
            return (
                isinstance(expr0, expr.Cast)
                and len(expr0.children) == 1
                and isinstance(expr0.children[0], expr.Len)
            )
        return False

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Select")
    def do_evaluate(
        cls,
        exprs: tuple[expr.NamedExpr, ...],
        should_broadcast: bool,  # noqa: FBT001
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # Handle any broadcasting
        columns = [e.evaluate(df) for e in exprs]
        if should_broadcast:
            columns = broadcast(*columns, stream=df.stream)
        return DataFrame(columns, stream=df.stream)

    def evaluate(
        self, *, cache: CSECache, timer: Timer | None, context: IRExecutionContext
    ) -> DataFrame:
        """
        Evaluate the Select node with special handling for fast count queries.

        Parameters
        ----------
        cache
            Mapping from cached node ids to constructed DataFrames.
            Used to implement evaluation of the `Cache` node.
        timer
            If not None, a Timer object to record timings for the
            evaluation of the node.
        context
            The execution context for the node.

        Returns
        -------
        DataFrame
            Result of evaluating this Select node. If the expression is a
            count over a parquet scan, returns a constant row count directly
            without evaluating the scan.

        Raises
        ------
        NotImplementedError
            If evaluation fails. Ideally this should not occur, since the
            translation phase should fail earlier.
        """
        if (
            isinstance(self.children[0], Scan)
            and Select._is_len_expr(self.exprs)
            and self.children[0].typ == "parquet"
            and self.children[0].predicate is None
        ):  # pragma: no cover
            stream = context.get_cuda_stream()
            scan = self.children[0]
            effective_rows = Scan._get_parquet_row_count_from_metadata(
                scan.paths,
                scan.skip_rows,
                scan.n_rows,
                scan.parquet_options,
                None,
            )
            dtype = DataType(pl.UInt32())
            col = Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(effective_rows, dtype.plc_type, stream=stream),
                    1,
                    stream=stream,
                ),
                name=self.exprs[0].name or "len",
                dtype=dtype,
            )
            return DataFrame([col], stream=stream)

        return super().evaluate(cache=cache, timer=timer, context=context)


class Reduce(IR):
    """
    Produce a new dataframe selecting given expressions from an input.

    This is a special case of :class:`Select` where all outputs are a single row.
    """

    __slots__ = ("exprs",)
    _non_child = ("schema", "exprs")
    _n_non_child_args = 1
    exprs: tuple[expr.NamedExpr, ...]
    """List of expressions to evaluate to form the new dataframe."""

    def __init__(
        self, schema: Schema, exprs: Sequence[expr.NamedExpr], df: IR
    ):  # pragma: no cover; polars doesn't emit this node yet
        self.schema = schema
        self.exprs = tuple(exprs)
        self.children = (df,)
        self._non_child_args = (self.exprs,)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Reduce")
    def do_evaluate(
        cls,
        exprs: tuple[expr.NamedExpr, ...],
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:  # pragma: no cover; not exposed by polars yet
        """Evaluate and return a dataframe."""
        columns = broadcast(*(e.evaluate(df) for e in exprs), stream=df.stream)
        assert all(column.size == 1 for column in columns)
        return DataFrame(columns, stream=df.stream)


class Rolling(IR):
    """Perform a (possibly grouped) rolling aggregation."""

    __slots__ = (
        "agg_requests",
        "closed_window",
        "following_ordinal",
        "index",
        "index_dtype",
        "keys",
        "preceding_ordinal",
        "zlice",
    )
    _non_child = (
        "schema",
        "index",
        "index_dtype",
        "preceding_ordinal",
        "following_ordinal",
        "closed_window",
        "keys",
        "agg_requests",
        "zlice",
    )
    _n_non_child_args = 8
    index: expr.NamedExpr
    """Column being rolled over."""
    index_dtype: plc.DataType
    """Datatype of the index column."""
    preceding_ordinal: int
    """Preceding window extent defining start of window as a host integer."""
    following_ordinal: int
    """Following window extent defining end of window as a host integer."""
    closed_window: ClosedInterval
    """Treatment of window endpoints."""
    keys: tuple[expr.NamedExpr, ...]
    """Grouping keys."""
    agg_requests: tuple[expr.NamedExpr, ...]
    """Aggregation expressions."""
    zlice: Zlice | None
    """Optional slice"""

    def __init__(
        self,
        schema: Schema,
        index: expr.NamedExpr,
        index_dtype: plc.DataType,
        preceding_ordinal: int,
        following_ordinal: int,
        closed_window: ClosedInterval,
        keys: Sequence[expr.NamedExpr],
        agg_requests: Sequence[expr.NamedExpr],
        zlice: Zlice | None,
        df: IR,
    ):
        self.schema = schema
        self.index = index
        self.index_dtype = index_dtype
        self.preceding_ordinal = preceding_ordinal
        self.following_ordinal = following_ordinal
        self.closed_window = closed_window
        self.keys = tuple(keys)
        self.agg_requests = tuple(agg_requests)
        if not all(
            plc.rolling.is_valid_rolling_aggregation(
                agg.value.dtype.plc_type, agg.value.agg_request
            )
            for agg in self.agg_requests
        ):
            raise NotImplementedError("Unsupported rolling aggregation")
        if any(
            agg.value.agg_request.kind() == plc.aggregation.Kind.COLLECT_LIST
            for agg in self.agg_requests
        ):
            raise NotImplementedError(
                "Incorrect handling of empty groups for list collection"
            )

        self.zlice = zlice
        self.children = (df,)
        self._non_child_args = (
            index,
            index_dtype,
            preceding_ordinal,
            following_ordinal,
            closed_window,
            keys,
            agg_requests,
            zlice,
        )

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Rolling")
    def do_evaluate(
        cls,
        index: expr.NamedExpr,
        index_dtype: plc.DataType,
        preceding_ordinal: int,
        following_ordinal: int,
        closed_window: ClosedInterval,
        keys_in: Sequence[expr.NamedExpr],
        aggs: Sequence[expr.NamedExpr],
        zlice: Zlice | None,
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        keys = broadcast(
            *(k.evaluate(df) for k in keys_in),
            target_length=df.num_rows,
            stream=df.stream,
        )
        orderby = index.evaluate(df)
        # Polars casts integral orderby to int64, but only for calculating window bounds
        if (
            plc.traits.is_integral(orderby.obj.type())
            and orderby.obj.type().id() != plc.TypeId.INT64
        ):
            orderby_obj = plc.unary.cast(
                orderby.obj, plc.DataType(plc.TypeId.INT64), stream=df.stream
            )
        else:
            orderby_obj = orderby.obj

        preceding_scalar, following_scalar = offsets_to_windows(
            index_dtype, preceding_ordinal, following_ordinal, stream=df.stream
        )

        preceding_window, following_window = range_window_bounds(
            preceding_scalar, following_scalar, closed_window
        )
        if orderby.obj.null_count() != 0:
            raise RuntimeError(
                f"Index column '{index.name}' in rolling may not contain nulls"
            )
        if len(keys_in) > 0:
            # Must always check sortedness
            table = plc.Table([*(k.obj for k in keys), orderby_obj])
            n = table.num_columns()
            if not plc.sorting.is_sorted(
                table,
                [plc.types.Order.ASCENDING] * n,
                [plc.types.NullOrder.BEFORE] * n,
                stream=df.stream,
            ):
                raise RuntimeError("Input for grouped rolling is not sorted")
        else:
            if not orderby.check_sorted(
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.BEFORE,
                stream=df.stream,
            ):
                raise RuntimeError(
                    f"Index column '{index.name}' in rolling is not sorted, please sort first"
                )
        values = plc.rolling.grouped_range_rolling_window(
            plc.Table([k.obj for k in keys]),
            orderby_obj,
            plc.types.Order.ASCENDING,  # Polars requires ascending orderby.
            plc.types.NullOrder.BEFORE,  # Doesn't matter, polars doesn't allow nulls in orderby
            preceding_window,
            following_window,
            [rolling.to_request(request.value, orderby, df) for request in aggs],
            stream=df.stream,
        )
        return DataFrame(
            itertools.chain(
                keys,
                [orderby],
                (
                    Column(col, name=request.name, dtype=request.value.dtype)
                    for col, request in zip(values.columns(), aggs, strict=True)
                ),
            ),
            stream=df.stream,
        ).slice(zlice)


def _has_struct_in_list(polars_type: pl.DataType) -> bool:
    if not isinstance(polars_type, pl.List):
        return False
    current = polars_type.inner
    while isinstance(current, pl.List):
        current = current.inner
    return isinstance(current, pl.Struct)


class GroupBy(IR):
    """Perform a groupby."""

    __slots__ = (
        "agg_requests",
        "keys",
        "maintain_order",
        "zlice",
    )
    _non_child = (
        "schema",
        "keys",
        "agg_requests",
        "maintain_order",
        "zlice",
    )
    _n_non_child_args = 5
    keys: tuple[expr.NamedExpr, ...]
    """Grouping keys."""
    agg_requests: tuple[expr.NamedExpr, ...]
    """Aggregation expressions."""
    maintain_order: bool
    """Preserve order in groupby."""
    zlice: Zlice | None
    """Optional slice to apply after grouping."""

    def __init__(
        self,
        schema: Schema,
        keys: Sequence[expr.NamedExpr],
        agg_requests: Sequence[expr.NamedExpr],
        maintain_order: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: IR,
    ):
        self.schema = schema
        self.keys = tuple(keys)
        for dtype in schema.values():
            if _has_struct_in_list(dtype.polars_type):
                raise NotImplementedError("Nested list[struct] types not supported")
        for request in agg_requests:
            expr = request.value
            if (
                POLARS_VERSION_LT_136
                and isinstance(expr, unary.UnaryFunction)
                and expr.name == "value_counts"
            ):
                raise NotImplementedError(
                    "value_counts is not supported in groupby"
                )  # pragma: no cover; Nested list[struct] types not supported
        self.agg_requests = tuple(agg_requests)
        self.maintain_order = maintain_order
        self.zlice = zlice
        self.children = (df,)
        self._non_child_args = (
            schema,
            self.keys,
            self.agg_requests,
            maintain_order,
            self.zlice,
        )

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="GroupBy")
    def do_evaluate(
        cls,
        schema: Schema,
        keys_in: Sequence[expr.NamedExpr],
        agg_requests: Sequence[expr.NamedExpr],
        maintain_order: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        keys = broadcast(
            *(k.evaluate(df) for k in keys_in),
            target_length=df.num_rows,
            stream=df.stream,
        )
        keys_are_sorted = (
            plc.types.Sorted.YES
            if all(k.is_sorted for k in keys)
            else plc.types.Sorted.NO
        )
        grouper = plc.groupby.GroupBy(
            plc.Table([k.obj for k in keys]),
            null_handling=plc.types.NullPolicy.INCLUDE,
            keys_are_sorted=keys_are_sorted,
            column_order=[k.order for k in keys],
            null_precedence=[k.null_order for k in keys],
        )
        requests = []
        names = []
        cast_to_schema = []
        for request in agg_requests:
            should_cast = False
            name = request.name
            value = request.value
            if isinstance(value, expr.Len):
                # A count aggregation, we need a column so use a key column
                col = keys[0].obj
            elif isinstance(value, expr.Agg):
                if value.name == "quantile":
                    child = value.children[0]
                else:
                    (child,) = value.children
                # libcudf will return int64 when summing integers
                # but the schema may be a lower bit width
                col = child.evaluate(df, context=ExecutionContext.GROUPBY).obj
                should_cast = value.name == "sum" and plc.traits.is_integral_not_bool(
                    col.type()
                )
            else:
                # Anything else, we pre-evaluate
                column = value.evaluate(df, context=ExecutionContext.GROUPBY)
                if column.size != keys[0].size:
                    column = broadcast(
                        column, target_length=keys[0].size, stream=df.stream
                    )[0]
                col = column.obj
            requests.append(plc.groupby.GroupByRequest(col, [value.agg_request]))
            names.append(name)
            cast_to_schema.append(should_cast)
        group_keys, raw_tables = grouper.aggregate(requests, stream=df.stream)
        results = [
            Column(column, name=name, dtype=schema[name])
            if not should_cast
            else Column(column, name=name, dtype=schema[name]).astype(
                schema[name], stream=df.stream
            )
            for name, column, should_cast in zip(
                names,
                itertools.chain.from_iterable(t.columns() for t in raw_tables),
                cast_to_schema,
                strict=True,
            )
        ]
        result_keys = [
            Column(grouped_key, name=key.name, dtype=key.dtype)
            for key, grouped_key in zip(keys, group_keys.columns(), strict=True)
        ]
        if keys_are_sorted:
            result_keys = [
                col.sorted_like(key) for col, key in zip(result_keys, keys, strict=True)
            ]
        broadcasted = broadcast(*result_keys, *results, stream=df.stream)
        # Handle order preservation of groups
        if maintain_order and not keys_are_sorted:
            # The order we want
            want = plc.stream_compaction.stable_distinct(
                plc.Table([k.obj for k in keys]),
                list(range(group_keys.num_columns())),
                plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
                stream=df.stream,
            )
            # The order we have
            have = plc.Table([key.obj for key in broadcasted[: len(keys)]])

            # We know an inner join is OK because by construction
            # want and have are permutations of each other.
            left_order, right_order = plc.join.inner_join(
                want, have, plc.types.NullEquality.EQUAL, stream=df.stream
            )
            # Now left_order is an arbitrary permutation of the ordering we
            # want, and right_order is a matching permutation of the ordering
            # we have. To get to the original ordering, we need
            # left_order == iota(nrows), with right_order permuted
            # appropriately. This can be obtained by sorting
            # right_order by left_order.
            (right_order,) = plc.sorting.sort_by_key(
                plc.Table([right_order]),
                plc.Table([left_order]),
                [plc.types.Order.ASCENDING],
                [plc.types.NullOrder.AFTER],
                stream=df.stream,
            ).columns()
            ordered_table = plc.copying.gather(
                plc.Table([col.obj for col in broadcasted]),
                right_order,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=df.stream,
            )
            broadcasted = [
                Column(reordered, name=old.name, dtype=old.dtype)
                for reordered, old in zip(
                    ordered_table.columns(), broadcasted, strict=True
                )
            ]
        return DataFrame(broadcasted, stream=df.stream).slice(zlice)


def _strip_predicate_casts(node: expr.Expr) -> expr.Expr:
    if isinstance(node, expr.Cast):
        (child,) = node.children
        child = _strip_predicate_casts(child)

        src = child.dtype
        dst = node.dtype

        if plc.traits.is_fixed_point(src.plc_type) or plc.traits.is_fixed_point(
            dst.plc_type
        ):
            return child

        if isinstance(child, expr.ColRef) and (
            (
                plc.traits.is_floating_point(src.plc_type)
                and plc.traits.is_floating_point(dst.plc_type)
            )
            or (
                plc.traits.is_integral(src.plc_type)
                and plc.traits.is_integral(dst.plc_type)
                and src.plc_type.id() == dst.plc_type.id()
            )
        ):
            return child

    if not node.children:
        return node
    return node.reconstruct([_strip_predicate_casts(child) for child in node.children])


def _add_cast(
    target: DataType,
    side: expr.ColRef,
    left_casts: dict[str, DataType],
    right_casts: dict[str, DataType],
) -> None:  # pragma: no cover
    (col,) = side.children
    assert isinstance(col, expr.Col)
    casts = (
        left_casts if side.table_ref == plc_expr.TableReference.LEFT else right_casts
    )
    casts[col.name] = target


def _align_decimal_binop_types(
    left_expr: expr.ColRef,
    right_expr: expr.ColRef,
    left_casts: dict[str, DataType],
    right_casts: dict[str, DataType],
) -> None:
    left_type, right_type = left_expr.dtype, right_expr.dtype

    if plc.traits.is_fixed_point(left_type.plc_type) and plc.traits.is_fixed_point(
        right_type.plc_type
    ):
        target = DataType.common_decimal_dtype(left_type, right_type)

        if (
            left_type.id() != target.id() or left_type.scale() != target.scale()
        ):  # pragma: no cover
            _add_cast(target, left_expr, left_casts, right_casts)

        if right_type.id() != target.id() or right_type.scale() != target.scale():
            _add_cast(target, right_expr, left_casts, right_casts)

    elif (
        plc.traits.is_fixed_point(left_type.plc_type)
        and plc.traits.is_floating_point(right_type.plc_type)
    ) or (
        plc.traits.is_fixed_point(right_type.plc_type)
        and plc.traits.is_floating_point(left_type.plc_type)
    ):  # pragma: no cover
        is_decimal_left = plc.traits.is_fixed_point(left_type.plc_type)
        decimal_expr, float_expr = (
            (left_expr, right_expr) if is_decimal_left else (right_expr, left_expr)
        )
        _add_cast(decimal_expr.dtype, float_expr, left_casts, right_casts)


def _collect_decimal_binop_casts(
    predicate: expr.Expr,
) -> tuple[dict[str, DataType], dict[str, DataType]]:
    left_casts: dict[str, DataType] = {}
    right_casts: dict[str, DataType] = {}

    def _walk(node: expr.Expr) -> None:
        if isinstance(node, expr.BinOp) and node.op in _BINOPS:
            left_expr, right_expr = node.children
            if isinstance(left_expr, expr.ColRef) and isinstance(
                right_expr, expr.ColRef
            ):
                _align_decimal_binop_types(
                    left_expr, right_expr, left_casts, right_casts
                )
        for child in node.children:
            _walk(child)

    _walk(predicate)
    return left_casts, right_casts


def _apply_casts(
    df: DataFrame, casts: dict[str, DataType]
) -> DataFrame:  # pragma: no cover
    if not casts:
        return df

    columns = []
    for col in df.columns:
        target = casts.get(col.name)
        if target is None:
            columns.append(Column(col.obj, dtype=col.dtype, name=col.name))
        else:
            casted = col.astype(target, stream=df.stream)
            columns.append(Column(casted.obj, dtype=casted.dtype, name=col.name))
    return DataFrame(columns, stream=df.stream)


class ConditionalJoin(IR):
    """A conditional inner join of two dataframes on a predicate."""

    class Predicate:
        """Serializable wrapper for a predicate expression."""

        predicate: expr.Expr
        ast: plc.expressions.Expression

        def __init__(self, predicate: expr.Expr):
            self.predicate = predicate
            stream = get_cuda_stream()
            ast_result = to_ast(predicate, stream=stream)
            stream.synchronize()
            if ast_result is None:
                raise NotImplementedError(
                    f"Conditional join with predicate {predicate}"
                )  # pragma: no cover; polars never delivers expressions we can't handle
            self.ast = ast_result

        def __reduce__(self) -> tuple[Any, ...]:
            """Pickle a Predicate object."""
            return (type(self), (self.predicate,))

    __slots__ = ("ast_predicate", "options", "predicate")
    _non_child = ("schema", "predicate", "options")
    _n_non_child_args = 2
    predicate: expr.Expr
    """Expression predicate to join on"""
    options: tuple[
        tuple[
            str,
            polars._expr_nodes.Operator | Iterable[polars._expr_nodes.Operator],
        ]
        | None,
        bool,
        Zlice | None,
        str,
        bool,
        Literal["none", "left", "right", "left_right", "right_left"],
    ]
    """
    tuple of options:
    - predicates: tuple of ir join type (eg. ie_join) and (In)Equality conditions
    - nulls_equal: do nulls compare equal?
    - slice: optional slice to perform after joining.
    - suffix: string suffix for right columns if names match
    - coalesce: should key columns be coalesced (only makes sense for outer joins)
    - maintain_order: which DataFrame row order to preserve, if any
    """

    def __init__(
        self, schema: Schema, predicate: expr.Expr, options: tuple, left: IR, right: IR
    ) -> None:
        self.schema = schema
        predicate = _strip_predicate_casts(predicate)
        self.predicate = predicate
        # options[0] is a tuple[str, Operator, ...]
        # The Operator class can't be pickled, but we don't use it anyway so
        # just throw that away
        if options[0] is not None:
            options = (None, *options[1:])

        self.options = options
        self.children = (left, right)
        predicate_wrapper = self.Predicate(predicate)
        _, nulls_equal, _zlice, _suffix, coalesce, maintain_order = self.options
        # Preconditions from polars
        assert not nulls_equal
        assert not coalesce
        assert maintain_order == "none"
        self._non_child_args = (predicate_wrapper, options)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="ConditionalJoin")
    def do_evaluate(
        cls,
        predicate_wrapper: Predicate,
        options: tuple,
        left: DataFrame,
        right: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        left_casts, right_casts = _collect_decimal_binop_casts(
            predicate_wrapper.predicate
        )
        left_on = _apply_casts(left, left_casts)
        right_on = _apply_casts(right, right_casts)
        with context.stream_ordered_after(left, right) as stream:
            _, _, zlice, suffix, _, _ = options

            lg, rg = plc.join.conditional_inner_join(
                left_on.table, right_on.table, predicate_wrapper.ast, stream=stream
            )
            left_result = DataFrame.from_table(
                plc.copying.gather(
                    left.table,
                    lg,
                    plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    stream=stream,
                ),
                left.column_names,
                left.dtypes,
                stream=stream,
            )
            right_result = DataFrame.from_table(
                plc.copying.gather(
                    right.table,
                    rg,
                    plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    stream=stream,
                ),
                right.column_names,
                right.dtypes,
                stream=stream,
            )
            right_result = right_result.rename_columns(
                {
                    name: f"{name}{suffix}"
                    for name in right.column_names
                    if name in left.column_names_set
                }
            )
            result = left_result.with_columns(right_result.columns, stream=stream)

        return result.slice(zlice)


class Join(IR):
    """A join of two dataframes."""

    __slots__ = ("left_on", "options", "right_on")
    _non_child = ("schema", "left_on", "right_on", "options")
    _n_non_child_args = 3
    left_on: tuple[expr.NamedExpr, ...]
    """List of expressions used as keys in the left frame."""
    right_on: tuple[expr.NamedExpr, ...]
    """List of expressions used as keys in the right frame."""
    options: tuple[
        Literal["Inner", "Left", "Right", "Full", "Semi", "Anti", "Cross"],
        bool,
        Zlice | None,
        str,
        bool,
        Literal["none", "left", "right", "left_right", "right_left"],
    ]
    """
    tuple of options:
    - how: join type
    - nulls_equal: do nulls compare equal?
    - slice: optional slice to perform after joining.
    - suffix: string suffix for right columns if names match
    - coalesce: should key columns be coalesced (only makes sense for outer joins)
    - maintain_order: which DataFrame row order to preserve, if any
    """

    SWAPPED_ORDER: ClassVar[
        dict[
            Literal["none", "left", "right", "left_right", "right_left"],
            Literal["none", "left", "right", "left_right", "right_left"],
        ]
    ] = {
        "none": "none",
        "left": "right",
        "right": "left",
        "left_right": "right_left",
        "right_left": "left_right",
    }

    def __init__(
        self,
        schema: Schema,
        left_on: Sequence[expr.NamedExpr],
        right_on: Sequence[expr.NamedExpr],
        options: Any,
        left: IR,
        right: IR,
    ):
        self.schema = schema
        self.left_on = tuple(left_on)
        self.right_on = tuple(right_on)
        self.options = options
        self.children = (left, right)
        self._non_child_args = (self.left_on, self.right_on, self.options)

    @staticmethod
    @cache
    def _joiners(
        how: Literal["Inner", "Left", "Right", "Full", "Semi", "Anti"],
    ) -> tuple[
        Callable, plc.copying.OutOfBoundsPolicy, plc.copying.OutOfBoundsPolicy | None
    ]:
        if how == "Inner":
            return (
                plc.join.inner_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
        elif how == "Left" or how == "Right":
            return (
                plc.join.left_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            )
        elif how == "Full":
            return (
                plc.join.full_join,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            )
        elif how == "Semi":
            return (
                plc.join.left_semi_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                None,
            )
        elif how == "Anti":
            return (
                plc.join.left_anti_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                None,
            )
        assert_never(how)

    @staticmethod
    def _reorder_maps(
        left_rows: int,
        lg: plc.Column,
        left_policy: plc.copying.OutOfBoundsPolicy,
        right_rows: int,
        rg: plc.Column,
        right_policy: plc.copying.OutOfBoundsPolicy,
        *,
        left_primary: bool = True,
        stream: Stream,
    ) -> tuple[plc.Column, ...]:
        """
        Reorder gather maps to satisfy polars join order restrictions.

        Parameters
        ----------
        left_rows
            Number of rows in left table
        lg
            Left gather map
        left_policy
            Nullify policy for left map
        right_rows
            Number of rows in right table
        rg
            Right gather map
        right_policy
            Nullify policy for right map
        left_primary
            Whether to preserve the left input row order first, and which
            input stream to use for the primary sort.
            Defaults to True.
        stream
            CUDA stream used for device memory operations and kernel launches.

        Returns
        -------
        tuple[plc.Column, ...]
            Reordered left and right gather maps.

        Notes
        -----
        When ``left_primary`` is True, the pair of gather maps is stably sorted by
        the original row order of the left side, breaking ties by the right side.
        And vice versa when ``left_primary`` is False.
        """
        init = plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream)
        step = plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream)

        (left_order_col,) = plc.copying.gather(
            plc.Table(
                [
                    plc.filling.sequence(
                        left_rows,
                        init,
                        step,
                        stream=stream,
                    )
                ]
            ),
            lg,
            left_policy,
            stream=stream,
        ).columns()
        (right_order_col,) = plc.copying.gather(
            plc.Table(
                [
                    plc.filling.sequence(
                        right_rows,
                        init,
                        step,
                        stream=stream,
                    )
                ]
            ),
            rg,
            right_policy,
            stream=stream,
        ).columns()

        keys = (
            plc.Table([left_order_col, right_order_col])
            if left_primary
            else plc.Table([right_order_col, left_order_col])
        )

        return plc.sorting.stable_sort_by_key(
            plc.Table([lg, rg]),
            keys,
            [plc.types.Order.ASCENDING, plc.types.Order.ASCENDING],
            [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
            stream=stream,
        ).columns()

    @staticmethod
    def _build_columns(
        columns: Iterable[plc.Column],
        template: Iterable[NamedColumn],
        *,
        left: bool = True,
        empty: bool = False,
        rename: Callable[[str], str] = lambda name: name,
        stream: Stream,
    ) -> list[Column]:
        if empty:
            return [
                Column(
                    plc.column_factories.make_empty_column(
                        col.dtype.plc_type, stream=stream
                    ),
                    col.dtype,
                    name=rename(col.name),
                )
                for col in template
            ]

        result = [
            Column(new, col.dtype, name=rename(col.name))
            for new, col in zip(columns, template, strict=True)
        ]

        if left:
            result = [
                col.sorted_like(orig)
                for col, orig in zip(result, template, strict=True)
            ]

        return result

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Join")
    def do_evaluate(
        cls,
        left_on_exprs: Sequence[expr.NamedExpr],
        right_on_exprs: Sequence[expr.NamedExpr],
        options: tuple[
            Literal["Inner", "Left", "Right", "Full", "Semi", "Anti", "Cross"],
            bool,
            Zlice | None,
            str,
            bool,
            Literal["none", "left", "right", "left_right", "right_left"],
        ],
        left: DataFrame,
        right: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        how, nulls_equal, zlice, suffix, coalesce, maintain_order = options
        if how == "Cross":
            with context.stream_ordered_after(left, right) as stream:
                # Separate implementation, since cross_join returns the
                # result, not the gather maps
                if right.num_rows == 0:
                    left_cols = Join._build_columns(
                        [], left.columns, empty=True, stream=stream
                    )
                    right_cols = Join._build_columns(
                        [],
                        right.columns,
                        left=False,
                        empty=True,
                        rename=lambda name: (
                            name
                            if name not in left.column_names_set
                            else f"{name}{suffix}"
                        ),
                        stream=stream,
                    )
                    return DataFrame([*left_cols, *right_cols], stream=stream).slice(
                        zlice
                    )
                else:
                    columns = plc.join.cross_join(
                        left.table, right.table, stream=stream
                    ).columns()
                    left_cols = Join._build_columns(
                        columns[: left.num_columns], left.columns, stream=stream
                    )
                    right_cols = Join._build_columns(
                        columns[left.num_columns :],
                        right.columns,
                        rename=lambda name: (
                            name
                            if name not in left.column_names_set
                            else f"{name}{suffix}"
                        ),
                        left=False,
                        stream=stream,
                    )
                    return DataFrame([*left_cols, *right_cols], stream=stream).slice(
                        zlice
                    )
        else:
            # how != "Cross"
            # TODO: Waiting on clarity based on https://github.com/pola-rs/polars/issues/17184
            left_on = DataFrame(
                broadcast(
                    *(e.evaluate(left) for e in left_on_exprs), stream=left.stream
                ),
                stream=left.stream,
            )
            right_on = DataFrame(
                broadcast(
                    *(e.evaluate(right) for e in right_on_exprs), stream=right.stream
                ),
                stream=right.stream,
            )
            with context.stream_ordered_after(left, right) as stream:
                null_equality = (
                    plc.types.NullEquality.EQUAL
                    if nulls_equal
                    else plc.types.NullEquality.UNEQUAL
                )
                join_fn, left_policy, right_policy = cls._joiners(how)
                if right_policy is None:
                    # Semi join
                    lg = join_fn(left_on.table, right_on.table, null_equality, stream)
                    table = plc.copying.gather(
                        left.table, lg, left_policy, stream=stream
                    )
                    return DataFrame.from_table(
                        table, left.column_names, left.dtypes, stream=stream
                    ).slice(zlice)
                else:
                    if how == "Right":
                        # Right join is a left join with the tables swapped
                        left, right = right, left
                        left_on, right_on = right_on, left_on
                        maintain_order = Join.SWAPPED_ORDER[maintain_order]
                    lg, rg = join_fn(
                        left_on.table, right_on.table, null_equality, stream=stream
                    )
                    if (
                        how in ("Inner", "Left", "Right", "Full")
                        and maintain_order != "none"
                    ):
                        lg, rg = cls._reorder_maps(
                            left.num_rows,
                            lg,
                            left_policy,
                            right.num_rows,
                            rg,
                            right_policy,
                            left_primary=maintain_order.startswith("left"),
                            stream=stream,
                        )
                    if coalesce:
                        if how == "Full":
                            # In this case, keys must be column references,
                            # possibly with dtype casting. We should use them in
                            # preference to the columns from the original tables.

                            # We need to specify `stream` here. We know that `{left,right}_on`
                            # is valid on `stream`, which is ordered after `{left,right}.stream`.
                            left = left.with_columns(
                                left_on.columns, replace_only=True, stream=stream
                            )
                            right = right.with_columns(
                                right_on.columns, replace_only=True, stream=stream
                            )
                        else:
                            right = right.discard_columns(right_on.column_names_set)
                    left = DataFrame.from_table(
                        plc.copying.gather(left.table, lg, left_policy, stream=stream),
                        left.column_names,
                        left.dtypes,
                        stream=stream,
                    )
                    right = DataFrame.from_table(
                        plc.copying.gather(
                            right.table, rg, right_policy, stream=stream
                        ),
                        right.column_names,
                        right.dtypes,
                        stream=stream,
                    )
                    if coalesce and how == "Full":
                        left = left.with_columns(
                            (
                                Column(
                                    plc.replace.replace_nulls(
                                        left_col.obj, right_col.obj, stream=stream
                                    ),
                                    name=left_col.name,
                                    dtype=left_col.dtype,
                                )
                                for left_col, right_col in zip(
                                    left.select_columns(left_on.column_names_set),
                                    right.select_columns(right_on.column_names_set),
                                    strict=True,
                                )
                            ),
                            replace_only=True,
                            stream=stream,
                        )
                        right = right.discard_columns(right_on.column_names_set)
                    if how == "Right":
                        # Undo the swap for right join before gluing together.
                        left, right = right, left
                    right = right.rename_columns(
                        {
                            name: f"{name}{suffix}"
                            for name in right.column_names
                            if name in left.column_names_set
                        }
                    )
                    return left.with_columns(right.columns, stream=stream).slice(zlice)


class HStack(IR):
    """Add new columns to a dataframe."""

    __slots__ = ("columns", "should_broadcast")
    _non_child = ("schema", "columns", "should_broadcast")
    _n_non_child_args = 2
    should_broadcast: bool
    """Should the resulting evaluated columns be broadcast to the same length."""

    def __init__(
        self,
        schema: Schema,
        columns: Sequence[expr.NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        df: IR,
    ):
        self.schema = schema
        self.columns = tuple(columns)
        self.should_broadcast = should_broadcast
        self._non_child_args = (self.columns, self.should_broadcast)
        self.children = (df,)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="HStack")
    def do_evaluate(
        cls,
        exprs: Sequence[expr.NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        columns = [c.evaluate(df) for c in exprs]
        if should_broadcast:
            columns = broadcast(
                *columns,
                target_length=df.num_rows if df.num_columns != 0 else None,
                stream=df.stream,
            )
        else:  # pragma: no cover; streaming rewrites HStack(False)
            # Polars ensures this is true, but let's make sure nothing
            # went wrong. In this case, the parent node is
            # guaranteed to be a Select which will take care of making
            # sure that everything is the same length. The result
            # table that might have mismatching column lengths will
            # never be turned into a pylibcudf Table with all columns
            # by the Select, which is why this is safe.
            assert all(e.name.startswith("__POLARS_CSER_0x") for e in exprs)
        return df.with_columns(columns, stream=df.stream)


class Distinct(IR):
    """Produce a new dataframe with distinct rows."""

    __slots__ = ("keep", "stable", "subset", "zlice")
    _non_child = ("schema", "keep", "subset", "zlice", "stable")
    _n_non_child_args = 4
    keep: plc.stream_compaction.DuplicateKeepOption
    """Which distinct value to keep."""
    subset: frozenset[str] | None
    """Which columns should be used to define distinctness. If None,
    then all columns are used."""
    zlice: Zlice | None
    """Optional slice to apply to the result."""
    stable: bool
    """Should the result maintain ordering."""

    def __init__(
        self,
        schema: Schema,
        keep: plc.stream_compaction.DuplicateKeepOption,
        subset: frozenset[str] | None,
        zlice: Zlice | None,
        stable: bool,  # noqa: FBT001
        df: IR,
    ):
        self.schema = schema
        self.keep = keep
        self.subset = subset
        self.zlice = zlice
        self.stable = stable
        self._non_child_args = (keep, subset, zlice, stable)
        self.children = (df,)

    _KEEP_MAP: ClassVar[dict[str, plc.stream_compaction.DuplicateKeepOption]] = {
        "first": plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        "last": plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        "none": plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
        "any": plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
    }

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Distinct")
    def do_evaluate(
        cls,
        keep: plc.stream_compaction.DuplicateKeepOption,
        subset: frozenset[str] | None,
        zlice: Zlice | None,
        stable: bool,  # noqa: FBT001
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        if subset is None:
            indices = list(range(df.num_columns))
            keys_sorted = all(c.is_sorted for c in df.column_map.values())
        else:
            indices = [i for i, k in enumerate(df.column_names) if k in subset]
            keys_sorted = all(df.column_map[name].is_sorted for name in subset)
        if keys_sorted:
            table = plc.stream_compaction.unique(
                df.table,
                indices,
                keep,
                plc.types.NullEquality.EQUAL,
                stream=df.stream,
            )
        else:
            distinct = (
                plc.stream_compaction.stable_distinct
                if stable
                else plc.stream_compaction.distinct
            )
            table = distinct(
                df.table,
                indices,
                keep,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
                df.stream,
            )
        # TODO: Is this sortedness setting correct
        result = DataFrame(
            [
                Column(new, name=old.name, dtype=old.dtype).sorted_like(old)
                for new, old in zip(table.columns(), df.columns, strict=True)
            ],
            stream=df.stream,
        )
        if keys_sorted or stable:
            result = result.sorted_like(df)
        return result.slice(zlice)


class Sort(IR):
    """Sort a dataframe."""

    __slots__ = ("by", "null_order", "order", "stable", "zlice")
    _non_child = ("schema", "by", "order", "null_order", "stable", "zlice")
    _n_non_child_args = 5
    by: tuple[expr.NamedExpr, ...]
    """Sort keys."""
    order: tuple[plc.types.Order, ...]
    """Sort order for each sort key."""
    null_order: tuple[plc.types.NullOrder, ...]
    """Null sorting location for each sort key."""
    stable: bool
    """Should the sort be stable?"""
    zlice: Zlice | None
    """Optional slice to apply to the result."""

    def __init__(
        self,
        schema: Schema,
        by: Sequence[expr.NamedExpr],
        order: Sequence[plc.types.Order],
        null_order: Sequence[plc.types.NullOrder],
        stable: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: IR,
    ):
        self.schema = schema
        self.by = tuple(by)
        self.order = tuple(order)
        self.null_order = tuple(null_order)
        self.stable = stable
        self.zlice = zlice
        self._non_child_args = (
            self.by,
            self.order,
            self.null_order,
            self.stable,
            self.zlice,
        )
        self.children = (df,)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Sort")
    def do_evaluate(
        cls,
        by: Sequence[expr.NamedExpr],
        order: Sequence[plc.types.Order],
        null_order: Sequence[plc.types.NullOrder],
        stable: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        sort_keys = broadcast(
            *(k.evaluate(df) for k in by), target_length=df.num_rows, stream=df.stream
        )
        do_sort = plc.sorting.stable_sort_by_key if stable else plc.sorting.sort_by_key
        table = do_sort(
            df.table,
            plc.Table([k.obj for k in sort_keys]),
            list(order),
            list(null_order),
            stream=df.stream,
        )
        result = DataFrame.from_table(
            table, df.column_names, df.dtypes, stream=df.stream
        )
        first_key = sort_keys[0]
        name = by[0].name
        first_key_in_result = (
            name in df.column_map and first_key.obj is df.column_map[name].obj
        )
        if first_key_in_result:
            result.column_map[name].set_sorted(
                is_sorted=plc.types.Sorted.YES, order=order[0], null_order=null_order[0]
            )
        return result.slice(zlice)


class Slice(IR):
    """Slice a dataframe."""

    __slots__ = ("length", "offset")
    _non_child = ("schema", "offset", "length")
    _n_non_child_args = 2
    offset: int
    """Start of the slice."""
    length: int | None
    """Length of the slice."""

    def __init__(self, schema: Schema, offset: int, length: int | None, df: IR):
        self.schema = schema
        self.offset = offset
        self.length = length
        self._non_child_args = (offset, length)
        self.children = (df,)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Slice")
    def do_evaluate(
        cls, offset: int, length: int, df: DataFrame, *, context: IRExecutionContext
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        return df.slice((offset, length))


class Filter(IR):
    """Filter a dataframe with a boolean mask."""

    __slots__ = ("mask",)
    _non_child = ("schema", "mask")
    _n_non_child_args = 1
    mask: expr.NamedExpr
    """Expression to produce the filter mask."""

    def __init__(self, schema: Schema, mask: expr.NamedExpr, df: IR):
        self.schema = schema
        self.mask = mask
        self._non_child_args = (mask,)
        self.children = (df,)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Filter")
    def do_evaluate(
        cls, mask_expr: expr.NamedExpr, df: DataFrame, *, context: IRExecutionContext
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        return apply_predicate(df, mask_expr)


class Projection(IR):
    """Select a subset of columns from a dataframe."""

    __slots__ = ()
    _non_child = ("schema",)
    _n_non_child_args = 1

    def __init__(self, schema: Schema, df: IR):
        self.schema = schema
        self._non_child_args = (schema,)
        self.children = (df,)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Projection")
    def do_evaluate(
        cls, schema: Schema, df: DataFrame, *, context: IRExecutionContext
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # This can reorder things.
        columns = broadcast(
            *(df.column_map[name] for name in schema),
            target_length=df.num_rows,
            stream=df.stream,
        )
        return DataFrame(
            columns,
            stream=df.stream,
            num_rows=df.num_rows if len(schema) == 0 else None,
        )


class MergeSorted(IR):
    """Merge sorted operation."""

    __slots__ = ("key",)
    _non_child = ("schema", "key")
    _n_non_child_args = 1
    key: str
    """Key that is sorted."""

    def __init__(self, schema: Schema, key: str, left: IR, right: IR):
        # Children must be Sort or Repartition(Sort).
        # The Repartition(Sort) case happens during fallback.
        left_sort_child = left if isinstance(left, Sort) else left.children[0]
        right_sort_child = right if isinstance(right, Sort) else right.children[0]
        assert isinstance(left_sort_child, Sort)
        assert isinstance(right_sort_child, Sort)
        assert left_sort_child.order == right_sort_child.order
        assert len(left.schema.keys()) <= len(right.schema.keys())
        self.schema = schema
        self.key = key
        self.children = (left, right)
        self._non_child_args = (key,)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="MergeSorted")
    def do_evaluate(
        cls, key: str, *dfs: DataFrame, context: IRExecutionContext
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        with context.stream_ordered_after(*dfs) as stream:
            left, right = dfs
            right = right.discard_columns(
                right.column_names_set - left.column_names_set
            )
            on_col_left = left.select_columns({key})[0]
            on_col_right = right.select_columns({key})[0]
            return DataFrame.from_table(
                plc.merge.merge(
                    [right.table, left.table],
                    [left.column_names.index(key), right.column_names.index(key)],
                    [on_col_left.order, on_col_right.order],
                    [on_col_left.null_order, on_col_right.null_order],
                    stream=stream,
                ),
                left.column_names,
                left.dtypes,
                stream=stream,
            )


class MapFunction(IR):
    """Apply some function to a dataframe."""

    __slots__ = ("name", "options")
    _non_child = ("schema", "name", "options")
    _n_non_child_args = 3
    name: str
    """Name of the function to apply"""
    options: Any
    """Arbitrary name-specific options"""

    _NAMES: ClassVar[frozenset[str]] = frozenset(
        [
            "rechunk",
            "rename",
            "explode",
            "unpivot",
            "row_index",
            "fast_count",
            "hint_sorted",
        ]
    )

    def __init__(self, schema: Schema, name: str, options: Any, df: IR):
        self.schema = schema
        self.name = name
        self.options = options
        self.children = (df,)
        if (
            self.name not in MapFunction._NAMES
        ):  # pragma: no cover; need more polars rust functions
            raise NotImplementedError(
                f"Unhandled map function {self.name}"
            )  # pragma: no cover
        if self.name == "explode":
            if POLARS_VERSION_LT_136 or len(self.options) == 1:
                (to_explode,) = self.options
            else:
                (to_explode, empty_as_null, keep_nulls) = self.options
                if (POLARS_VERSION_LT_136 and not empty_as_null) or not keep_nulls:
                    raise NotImplementedError(
                        "Explode an empty or null list/array"
                    )  # pragma: no cover
            if len(to_explode) > 1:
                # TODO: straightforward, but need to error check
                # polars requires that all to-explode columns have the
                # same sub-shapes
                raise NotImplementedError("Explode with more than one column")
            self.options = (tuple(to_explode),)
        elif self.name == "unpivot":
            indices, pivotees, variable_name, value_name = self.options
            value_name = "value" if value_name is None else value_name
            variable_name = "variable" if variable_name is None else variable_name
            if POLARS_VERSION_LT_136 and len(pivotees) == 0:  # pragma: no cover
                index = frozenset(indices)
                pivotees = [name for name in df.schema if name not in index]
            if not all(
                dtypes.can_cast(df.schema[p].plc_type, self.schema[value_name].plc_type)
                for p in pivotees
            ):
                raise NotImplementedError(
                    "Unpivot cannot cast all input columns to "
                    f"{self.schema[value_name].id()}"
                )  # pragma: no cover
            self.options = (
                tuple(indices),
                tuple(pivotees),
                variable_name,
                value_name,
            )
        elif self.name == "row_index":
            col_name, offset = options
            if POLARS_VERSION_LT_138 and col_name in df.schema:  # pragma: no cover
                raise NotImplementedError("Duplicate row index name")
            self.options = (col_name, offset)
        elif self.name == "fast_count":
            # TODO: Remove this once all scan types support projections
            # using Select + Len. Currently, CSV is the only format that
            # uses the legacy MapFunction(FastCount) path because it is
            # faster than the new-streaming path for large files.
            # See https://github.com/pola-rs/polars/pull/22363#issue-3010224808
            raise NotImplementedError(
                "Fast count unsupported for CSV scans"
            )  # pragma: no cover
        elif (
            self.name == "hint_sorted"
        ):  # pragma: no cover; polars prunes hints in some cases
            raise NotImplementedError("Hint sorted unsupported")
        self._non_child_args = (schema, name, self.options)

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of the node.

        The options dictionaries are serialised for hashing purposes
        as json strings.
        """
        return (
            type(self),
            self.name,
            json.dumps(self.options),
            tuple(self.schema.items()),
            self._ctor_arguments(self.children)[1:],
        )

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="MapFunction")
    def do_evaluate(
        cls,
        schema: Schema,
        name: str,
        options: Any,
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        if name == "rechunk":
            # No-op in our data model
            # Don't think this appears in a plan tree from python
            return df  # pragma: no cover
        elif name == "explode":
            ((to_explode,),) = options
            index = df.column_names.index(to_explode)
            subset = df.column_names_set - {to_explode}
            return DataFrame.from_table(
                plc.lists.explode_outer(df.table, index, stream=df.stream),
                df.column_names,
                df.dtypes,
                stream=df.stream,
            ).sorted_like(df, subset=subset)
        elif name == "unpivot":
            (
                indices,
                pivotees,
                variable_name,
                value_name,
            ) = options
            npiv = len(pivotees)
            selected = df.select(indices)
            index_columns = [
                Column(tiled, name=name, dtype=old.dtype)
                for tiled, name, old in zip(
                    plc.reshape.tile(selected.table, npiv, stream=df.stream).columns(),
                    indices,
                    selected.columns,
                    strict=True,
                )
            ]
            (variable_column,) = plc.filling.repeat(
                plc.Table(
                    [
                        plc.Column.from_arrow(
                            pl.Series(
                                values=pivotees, dtype=schema[variable_name].polars_type
                            ),
                            stream=df.stream,
                        )
                    ]
                ),
                df.num_rows,
                stream=df.stream,
            ).columns()
            value_column = plc.concatenate.concatenate(
                [
                    df.column_map[pivotee]
                    .astype(schema[value_name], stream=df.stream)
                    .obj
                    for pivotee in pivotees
                ],
                stream=df.stream,
            )
            return DataFrame(
                [
                    *index_columns,
                    Column(
                        variable_column, name=variable_name, dtype=schema[variable_name]
                    ),
                    Column(value_column, name=value_name, dtype=schema[value_name]),
                ],
                stream=df.stream,
            )
        elif name == "row_index":
            col_name, offset = options
            dtype = schema[col_name]
            step = plc.Scalar.from_py(1, dtype.plc_type, stream=df.stream)
            init = plc.Scalar.from_py(offset, dtype.plc_type, stream=df.stream)
            index_col = Column(
                plc.filling.sequence(df.num_rows, init, step, stream=df.stream),
                is_sorted=plc.types.Sorted.YES,
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.AFTER,
                name=col_name,
                dtype=dtype,
            )
            return DataFrame([index_col, *df.columns], stream=df.stream)
        else:
            raise AssertionError("Should never be reached")  # pragma: no cover


class Union(IR):
    """Concatenate dataframes vertically."""

    __slots__ = ("maintain_order", "zlice")
    _non_child = ("schema", "zlice", "maintain_order")
    _n_non_child_args = 1
    maintain_order: bool
    """Whether row order must be preserved."""
    zlice: Zlice | None
    """Optional slice to apply to the result."""

    def __init__(
        self,
        schema: Schema,
        zlice: Zlice | None,
        maintain_order: bool,  # noqa: FBT001
        *children: IR,
    ):
        self.schema = schema
        self.zlice = zlice
        self.maintain_order = maintain_order
        self._non_child_args = (zlice,)
        self.children = children
        schema = self.children[0].schema

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Union")
    def do_evaluate(
        cls, zlice: Zlice | None, *dfs: DataFrame, context: IRExecutionContext
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # TODO: only evaluate what we need if we have a slice?
        return concatenate_chunks(dfs, context=context).slice(zlice)


class HConcat(IR):
    """Concatenate dataframes horizontally."""

    __slots__ = ("should_broadcast", "strict")
    _non_child = ("schema", "should_broadcast", "strict")
    _n_non_child_args = 3

    def __init__(
        self,
        schema: Schema,
        should_broadcast: bool,  # noqa: FBT001
        strict: bool,  # noqa: FBT001
        *children: IR,
    ):
        self.schema = schema
        self.should_broadcast = should_broadcast
        self.strict = strict
        self._non_child_args = (schema, should_broadcast, strict)
        self.children = children

    @staticmethod
    def _extend_with_nulls(
        table: plc.Table, *, nrows: int, stream: Stream
    ) -> plc.Table:
        """
        Extend a table with nulls.

        Parameters
        ----------
        table
            Table to extend
        nrows
            Number of additional rows
        stream
            CUDA stream used for device memory operations and kernel launches

        Returns
        -------
        New pylibcudf table.
        """
        return plc.concatenate.concatenate(
            [
                table,
                plc.Table(
                    [
                        plc.Column.all_null_like(column, nrows, stream=stream)
                        for column in table.columns()
                    ]
                ),
            ],
            stream=stream,
        )

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="HConcat")
    def do_evaluate(
        cls,
        schema: Schema,
        should_broadcast: bool,  # noqa: FBT001
        strict: bool,  # noqa: FBT001
        *dfs: DataFrame,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        with context.stream_ordered_after(*dfs) as stream:
            # Special should_broadcast case.
            # Used to recombine decomposed expressions
            if should_broadcast:
                bcols = broadcast(
                    *itertools.chain.from_iterable(df.columns for df in dfs),
                    stream=stream,
                )
                by_name = {c.name: c for c in bcols}
                ordered = [by_name[name] for name in schema]
                result = DataFrame(ordered, stream=stream)
            else:
                max_rows = max(df.num_rows for df in dfs)
                if strict and any(df.num_rows != max_rows for df in dfs):
                    heights = [df.num_rows for df in dfs]
                    msg = (
                        f"cannot concat DataFrames horizontally"
                        f" with strict=True: height mismatch {heights}"
                    )
                    raise pl.exceptions.ShapeError(msg)
                # Horizontal concatenation extends shorter tables with nulls
                result = DataFrame(
                    itertools.chain.from_iterable(
                        df.columns
                        for df in (
                            df
                            if df.num_rows == max_rows
                            else DataFrame.from_table(
                                cls._extend_with_nulls(
                                    df.table,
                                    nrows=max_rows - df.num_rows,
                                    stream=stream,
                                ),
                                df.column_names,
                                df.dtypes,
                                stream=stream,
                            )
                            for df in dfs
                        )
                    ),
                    stream=stream,
                )

        return result


class Empty(IR):
    """Represents an empty DataFrame with a known schema."""

    __slots__ = ("schema",)
    _non_child = ("schema",)
    _n_non_child_args = 1

    def __init__(self, schema: Schema):
        self.schema = schema
        self._non_child_args = (schema,)
        self.children = ()

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="Empty")
    def do_evaluate(
        cls, schema: Schema, *, context: IRExecutionContext
    ) -> DataFrame:  # pragma: no cover
        """Evaluate and return a dataframe."""
        stream = context.get_cuda_stream()
        return DataFrame(
            [
                Column(
                    plc.column_factories.make_empty_column(
                        dtype.plc_type, stream=stream
                    ),
                    dtype=dtype,
                    name=name,
                )
                for name, dtype in schema.items()
            ],
            stream=stream,
        )
