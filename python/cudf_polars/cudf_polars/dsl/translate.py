# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate polars IR representation to ours."""

from __future__ import annotations

import functools
import json
from contextlib import AbstractContextManager, nullcontext
from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never

import polars as pl

# polars.polars is not a part of the public API,
# so we cannot rely on importing it directly
# See https://github.com/pola-rs/polars/issues/24826
from polars import polars as plrs  # type: ignore[attr-defined]

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.to_ast import insert_colrefs
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.aggregations import decompose_single_agg
from cudf_polars.dsl.utils.groupby import rewrite_groupby
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.replace import replace
from cudf_polars.dsl.utils.rolling import rewrite_rolling
from cudf_polars.typing import Schema
from cudf_polars.utils import config, sorting
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_136,
    POLARS_VERSION_LT_138,
    POLARS_VERSION_LT_139,
    POLARS_VERSION_LT_140,
    POLARS_VERSION_LT_141,
    POLARS_VERSION_LT_142,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from polars import GPUEngine

    from cudf_polars.typing import NodeTraverser, Slice as Zlice

_HAS_ROLLING_FUNCTION = hasattr(plrs._expr_nodes, "RollingFunction")

__all__ = ["Translator", "translate_named_expr"]


def _align_decimal_float_for_comparison(
    *operands: expr.Expr,
) -> tuple[expr.Expr, ...]:
    """
    Cast Decimal operands to Float64 when mixed with Float operands.

    libcudf does not produce correct results for Decimal vs Float
    comparisons. This matches Polars' supertype resolution where
    Decimal + Float yields Float64.
    """
    has_decimal = any(plc.traits.is_fixed_point(op.dtype.plc_type) for op in operands)
    has_float = any(plc.traits.is_floating_point(op.dtype.plc_type) for op in operands)
    if has_decimal and has_float:
        f64 = DataType(pl.Float64())
        return tuple(
            expr.Cast(f64, False, op)  # noqa: FBT003
            if plc.traits.is_fixed_point(op.dtype.plc_type)
            else op
            for op in operands
        )
    return operands


def _strip_file_uri(path: str) -> str:
    # file:///foo/path is treated as a local path. Polars
    # rejects file:// URIs with any non-empty host, so
    # we don't need to handle file://host/foo/path paths
    prefix = "file://"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _check_compression(data: bytes) -> str | None:
    # Vendored from Polars' SupportedCompression::check in
    # polars-io/src/utils/compression.rs

    # TODO: Make polars hand us the compression info in the plan
    if len(data) < 4:
        return None

    # See https://en.wikipedia.org/wiki/List_of_file_signatures
    # for the specific hex signatures.
    if data[:2] == b"\x1f\x8b":
        return "gzip"

    if data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
        return "zlib"

    if data[:4] == b"\x28\xb5\x2f\xfd":
        return "zstd"

    return None


def _read_file_bytes(path: Path, num_bytes: int = 4) -> bytes:
    with path.open("rb") as f:
        return f.read(num_bytes)


def _unsupported_fill_over_window(value: expr.Expr) -> bool:
    """
    Check if a fill_null_with_strategy over a window function is unsupported.

    The only supported pattern is fill_null_with_strategy(cum_sum(...)) where
    cum_sum is the direct and only windowed child.
    """
    if not (
        isinstance(value, expr.UnaryFunction)
        and value.name == "fill_null_with_strategy"
    ):
        return False
    windowed = [
        node
        for node in traversal([value])
        if isinstance(node, expr.UnaryFunction)
        and node.name in {"rank", "cum_sum", "shift", "shift_and_fill"}
    ]
    if not windowed:
        return False
    child = value.children[0]
    return not (
        len(windowed) == 1
        and windowed[0] is child
        and isinstance(child, expr.UnaryFunction)
        and child.name == "cum_sum"
    )


class Translator:
    """
    Translates polars-internal IR nodes and expressions to our representation.

    Parameters
    ----------
    visitor
        Polars NodeTraverser object
    engine
        GPU engine configuration.
    """

    def __init__(self, visitor: NodeTraverser, engine: GPUEngine):
        self.visitor = visitor
        self.config_options = config.ConfigOptions.from_polars_engine(engine)
        self.errors: list[Exception] = []
        self._cache_nodes: dict[int, ir.Cache] = {}
        self._expr_context: ExecutionContext = ExecutionContext.FRAME
        self._internal_name_gen: Generator[str, None, None] | None = None

    def translate_ir(self, *, n: int | None = None) -> ir.IR:
        """
        Translate a polars-internal IR node to our representation.

        Parameters
        ----------
        visitor
            Polars NodeTraverser object
        n
            Optional node to start traversing from, if not provided uses
            current polars-internal node.

        Returns
        -------
        Translated IR object

        Raises
        ------
        NotImplementedError
            If the version of Polars IR is unsupported.

        Notes
        -----
        Any expression nodes that cannot be translated are replaced by
        :class:`expr.ErrorNode` nodes and collected in the the `errors` attribute.
        After translation is complete, this list of errors should be inspected
        to determine if the query is supported.
        """
        ctx: AbstractContextManager[None] = (
            set_node(self.visitor, n) if n is not None else noop_context
        )
        # IR is versioned with major.minor, minor is bumped for backwards
        # compatible changes (e.g. adding new nodes), major is bumped for
        # incompatible changes (e.g. renaming nodes).
        if (version := self.visitor.version()) >= (14, 4):
            e = NotImplementedError(
                f"No support for polars IR {version=}"
            )  # pragma: no cover; no such version for now.
            self.errors.append(e)  # pragma: no cover
            raise e  # pragma: no cover

        with ctx:
            polars_schema = self.visitor.get_schema()
            try:
                schema = {k: DataType(v) for k, v in polars_schema.items()}
            except Exception as e:
                self.errors.append(NotImplementedError(str(e)))
                return ir.ErrorNode({}, str(e))
            try:
                node = self.visitor.view_current_node()
            except Exception as e:
                self.errors.append(e)
                return ir.ErrorNode(schema, str(e))
            try:
                result = _translate_ir(node, self, schema)
            except Exception as e:
                self.errors.append(e)
                return ir.ErrorNode(schema, str(e))
            if any(
                isinstance(dtype, pl.Null)
                for dtype in pl.datatypes.unpack_dtypes(*polars_schema.values())
            ):
                error = NotImplementedError(
                    f"No GPU support for {result} with Null column dtype."
                )
                self.errors.append(error)
                return ir.ErrorNode(schema, str(error))

            return result

    def unsupported_operations_error(self) -> NotImplementedError | None:
        """
        Build an error describing unsupported operations during translation.

        Returns
        -------
        A `NotImplementedError` whose message (``args[0]``) is safe to surface
        to users and whose second argument contains the deduplicated underlying
        errors, or ``None`` if no translation errors were recorded.
        """
        if not self.errors:
            return None
        unique_errors = sorted({str(e): e for e in self.errors}.values(), key=str)
        # TODO: Display these errors in user-friendly way, tracked in
        # https://github.com/NVIDIA/cudf/issues/17051
        formatted_errors = "\n".join(
            f"- {e.__class__.__name__}: {e}" for e in unique_errors
        )
        message = (
            "Query execution with GPU not possible: unsupported operations."
            f"\nThe errors were:\n{formatted_errors}"
        )
        return NotImplementedError(message, unique_errors)

    def translate_expr(self, *, n: int, schema: Schema) -> expr.Expr:
        """
        Translate a polars-internal expression IR into our representation.

        Parameters
        ----------
        n
            Node to translate, an integer referencing a polars internal node.
        schema
            Schema of the IR node this expression uses as evaluation context.

        Returns
        -------
        Translated IR object.

        Notes
        -----
        Any expression nodes that cannot be translated are replaced by
        :class:`expr.ErrorExpr` nodes and collected in the the `errors` attribute.
        After translation is complete, this list of errors should be inspected
        to determine if the query is supported.
        """
        node = self.visitor.view_expression(n)
        dtype = DataType(self.visitor.get_dtype(n))
        try:
            return _translate_expr(node, self, dtype, schema)
        except Exception as e:
            self.errors.append(e)
            return expr.ErrorExpr(dtype, str(e))


class set_node(AbstractContextManager[None]):
    """
    Run a block with current node set in the visitor.

    Parameters
    ----------
    visitor
        The internal Rust visitor object
    n
        The node to set as the current root.

    Notes
    -----
    This is useful for translating expressions with a given node
    active, restoring the node when the block exits.
    """

    __slots__ = ("n", "visitor")
    visitor: NodeTraverser

    n: int

    def __init__(self, visitor: NodeTraverser, n: int) -> None:
        self.visitor = visitor
        self.n = n

    def __enter__(self) -> None:
        n = self.visitor.get_node()
        self.visitor.set_node(self.n)
        self.n = n

    def __exit__(self, *args: Any) -> None:
        self.visitor.set_node(self.n)


noop_context: nullcontext[None] = nullcontext()


class set_expr_context(AbstractContextManager[None]):
    __slots__ = ("_prev", "ctx", "translator")

    def __init__(self, translator: Translator, ctx: ExecutionContext) -> None:
        self.translator = translator
        self.ctx = ctx
        self._prev: ExecutionContext | None = None

    def __enter__(self) -> None:
        self._prev = self.translator._expr_context
        self.translator._expr_context = self.ctx

    def __exit__(self, *args: Any) -> None:
        assert self._prev is not None
        self.translator._expr_context = self._prev


def _drop_dyn_pred_hints(
    translator: Translator, n: int, schema: Schema
) -> expr.Expr | None:
    try:
        node = translator.visitor.view_expression(n)
    except Exception as e:  # pragma: no cover
        if str(e) == "dynamic_pred":
            return None
        raise
    if (
        not POLARS_VERSION_LT_142
        and isinstance(node, plrs._expr_nodes.Function)
        and node.function_data[0] == "dynamic_pred"
    ):
        return None
    if isinstance(node, plrs._expr_nodes.BinaryExpr) and node.op in (
        plrs._expr_nodes.Operator.And,
        plrs._expr_nodes.Operator.LogicalAnd,
    ):
        left = _drop_dyn_pred_hints(translator, node.left, schema)
        right = _drop_dyn_pred_hints(translator, node.right, schema)
        # Which branch is taken depends on where the dynamic predicate lands in
        # the AND tree, which is nondeterministic.
        if left is None:  # pragma: no cover
            return right
        if right is None:  # pragma: no cover
            return left
        return expr.BinOp(
            DataType(translator.visitor.get_dtype(n)),
            expr.BinOp._MAPPING[node.op],
            left,
            right,
        )
    return translator.translate_expr(n=n, schema=schema)


def translate_predicate(
    translator: Translator, *, n: int, output_name: str, schema: Schema
) -> expr.NamedExpr | None:
    """Translate predicate, dropping Polars' dynamic predicate hints."""
    mask = _drop_dyn_pred_hints(translator, n, schema)
    if mask is None:
        return None
    return expr.NamedExpr(output_name, mask)


class set_internal_name_gen(AbstractContextManager[None]):
    """Share one internal-name generator across sibling expression translations."""

    __slots__ = ("_prev", "schema", "translator")

    def __init__(self, translator: Translator, schema: Schema) -> None:
        self.translator = translator
        self.schema = schema
        self._prev: Generator[str, None, None] | None = None

    def __enter__(self) -> None:
        self._prev = self.translator._internal_name_gen
        self.translator._internal_name_gen = unique_names(self.schema)

    def __exit__(self, *args: Any) -> None:
        self.translator._internal_name_gen = self._prev


@singledispatch
def _translate_ir(node: Any, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_ir.register
def _(node: plrs._ir_nodes.PythonScan, translator: Translator, schema: Schema) -> ir.IR:
    scan_fn, with_columns, source_type, raw_predicate, nrows = node.options
    if source_type != "io_plugin":
        # cudf-polars supports register_io_source plugins (and pl.defer), which
        # report source_type "io_plugin". Other sources, notably pyarrow-dataset
        # scans ("pyarrow"), are not supported and fall back to the CPU engine.
        raise NotImplementedError(
            f"Unsupported PythonScan source type: {source_type!r}"
        )
    if nrows is not None:
        # A global row limit cannot be enforced independently per rank; tracked
        # in https://github.com/NVIDIA/cudf/issues/22918.
        raise NotImplementedError(
            "A row limit (head/limit) on a PythonScan source is not supported."
        )
    # Reused sources share the same ``scan_fn`` object, so capture the Polars
    # node index to keep occurrences distinct in our IR identity (hash/eq).
    # The index is deterministic across ranks.
    ir_node_index = translator.visitor.get_node()
    options = (scan_fn, with_columns, source_type, ir_node_index)
    if raw_predicate is None:
        predicate = None
    elif isinstance(raw_predicate, tuple) and raw_predicate[0] == "polars":
        predicate = translate_predicate(
            translator, n=raw_predicate[1], output_name="_predicate", schema=schema
        )
    else:  # pragma: no cover
        # The only other form is a pyarrow predicate, ("pyarrow", ...), which
        # Polars only emits for pyarrow-dataset scans, never for an
        # ``register_io_source`` plugin.
        assert_never(raw_predicate)
    return ir.PythonScan(schema, options, predicate)


@_translate_ir.register
def _(node: plrs._ir_nodes.Scan, translator: Translator, schema: Schema) -> ir.IR:
    typ, *options = node.scan_type
    paths = [_strip_file_uri(p) for p in node.paths]
    # Polars can produce a Scan with an empty ``node.paths`` (eg. the native
    # Iceberg reader on a table with no data files yet). In this case, polars returns an
    # empty DataFrame with the declared schema. Mirror that here by
    # replacing the Scan with an Empty IR node.
    if not paths:  # pragma: no cover
        return ir.Empty(schema)
    if typ == "ndjson":
        (reader_options,) = map(json.loads, options)
        cloud_options = None
    else:
        reader_options, cloud_options = map(json.loads, options)
    file_options = node.file_options
    with_columns = file_options.with_columns
    row_index = file_options.row_index
    include_file_paths = file_options.include_file_paths
    deletion_files = file_options.deletion_files
    if deletion_files:  # pragma: no cover
        raise NotImplementedError(
            "Iceberg format is not supported in cudf-polars. Furthermore, row-level deletions are not supported."
        )  # pragma: no cover
    if not POLARS_VERSION_LT_142 and node.hive_parts is not None:
        raise NotImplementedError("Hive-partitioned scans are not supported")
    config_options = translator.config_options
    parquet_options = config_options.parquet_options

    pre_slice = file_options.n_rows
    if pre_slice is None:
        n_rows = -1
        skip_rows = 0
    else:
        skip_rows, n_rows = pre_slice
        if (n_rows == 2**32 - 1) or (n_rows == 2**64 - 1):
            # Polars translates slice(10, None) -> (10, u32/64max)
            n_rows = -1

    # TODO: Get compression info from Polars plan
    if typ in ("csv", "ndjson"):
        for p in paths:
            if plc.io.SourceInfo._is_remote_uri(p):
                continue  # pragma: no cover
            data = _read_file_bytes(Path(p))
            compression = _check_compression(data)
            if compression is not None:
                raise NotImplementedError(
                    f"Reading compressed {typ.upper()} files is not supported."
                )

    return ir.Scan(
        schema,
        typ,
        reader_options,
        cloud_options,
        paths,
        with_columns,
        skip_rows,
        n_rows,
        row_index,
        include_file_paths,
        (
            None
            if node.predicate is None
            else translate_predicate(
                translator,
                n=node.predicate.node,
                output_name=node.predicate.output_name,
                schema=schema,
            )
        ),
        parquet_options,
        cached_parquet_info=None,
    )


@_translate_ir.register
def _(node: plrs._ir_nodes.Cache, translator: Translator, schema: Schema) -> ir.IR:
    # Make sure Cache nodes with the same id_
    # are actually the same object.
    if node.id_ not in translator._cache_nodes:
        translator._cache_nodes[node.id_] = ir.Cache(
            schema,
            node.id_,
            None,
            translator.translate_ir(n=node.input),
        )
    return translator._cache_nodes[node.id_]


@_translate_ir.register
def _(
    node: plrs._ir_nodes.DataFrameScan, translator: Translator, schema: Schema
) -> ir.IR:
    return ir.DataFrameScan(
        schema,
        node.df,
        node.projection,
    )


@_translate_ir.register
def _(node: plrs._ir_nodes.Select, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        with set_internal_name_gen(translator, inp.schema):
            exprs = [
                translate_named_expr(translator, n=e, schema=inp.schema)
                for e in node.expr
            ]
    return ir.Select(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(node: plrs._ir_nodes.GroupBy, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        keys = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.keys
        ]
        with set_expr_context(translator, ExecutionContext.GROUPBY):
            original_aggs = [
                translate_named_expr(translator, n=e, schema=inp.schema)
                for e in node.aggs
            ]
    is_rolling = node.options.rolling is not None
    is_dynamic = node.options.dynamic is not None
    if is_dynamic:
        raise NotImplementedError("group_by_dynamic")
    elif is_rolling:
        return rewrite_rolling(
            node.options, schema, keys, original_aggs, translator.config_options, inp
        )
    else:
        # TODO: Investigate whether polars can raise ahead of time
        # polars >= 1.40 rewrites an empty-key group_by into a Select, so an
        # empty-key group_by only reaches here on older polars; the raise is
        # dead on the latest polars used for coverage.
        if POLARS_VERSION_LT_140 and not keys:
            raise NotImplementedError(  # pragma: no cover
                "at least one key is required in a group_by operation"
            )
        return rewrite_groupby(node, schema, keys, original_aggs, inp)


@_translate_ir.register
def _(node: plrs._ir_nodes.Join, translator: Translator, schema: Schema) -> ir.IR:
    # Join key dtypes are dependent on the schema of the left and
    # right inputs, so these must be translated with the relevant
    # input active.
    with set_node(translator.visitor, node.input_left):
        inp_left = translator.translate_ir(n=None)
        left_on = [
            translate_named_expr(translator, n=e, schema=inp_left.schema)
            for e in node.left_on
        ]
    with set_node(translator.visitor, node.input_right):
        inp_right = translator.translate_ir(n=None)
        right_on = [
            translate_named_expr(translator, n=e, schema=inp_right.schema)
            for e in node.right_on
        ]

    if (how := node.options[0]) in {
        "Inner",
        "Left",
        "Right",
        "Full",
        "Cross",
        "Semi",
        "Anti",
    }:
        return ir.Join(
            schema,
            left_on,
            right_on,
            node.options,
            inp_left,
            inp_right,
        )
    else:
        how, op1, op2 = node.options[0]
        if how != "IEJoin":
            raise NotImplementedError(
                f"Unsupported join type {how}"
            )  # pragma: no cover; asof joins not yet exposed
        if op2 is None:
            ops = [op1]
        else:
            ops = [op1, op2]

        dtype = DataType(pl.datatypes.Boolean())
        predicate = functools.reduce(
            functools.partial(
                expr.BinOp, dtype, plc.binaryop.BinaryOperator.LOGICAL_AND
            ),
            (
                expr.BinOp(
                    dtype,
                    expr.BinOp._MAPPING[op],
                    insert_colrefs(
                        left_ne.value,
                        table_ref=plc.expressions.TableReference.LEFT,
                        name_to_index={
                            name: i for i, name in enumerate(inp_left.schema)
                        },
                    ),
                    insert_colrefs(
                        right_ne.value,
                        table_ref=plc.expressions.TableReference.RIGHT,
                        name_to_index={
                            name: i for i, name in enumerate(inp_right.schema)
                        },
                    ),
                )
                for op, left_ne, right_ne in zip(ops, left_on, right_on, strict=True)
            ),
        )

        return ir.ConditionalJoin(schema, predicate, node.options, inp_left, inp_right)


@_translate_ir.register
def _(node: plrs._ir_nodes.HStack, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        with set_internal_name_gen(translator, inp.schema):
            exprs = [
                translate_named_expr(translator, n=e, schema=inp.schema)
                for e in node.exprs
            ]
    return ir.HStack(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(
    node: plrs._ir_nodes.Reduce, translator: Translator, schema: Schema
) -> ir.IR:  # pragma: no cover; polars doesn't emit this node yet
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.expr
        ]
    return ir.Reduce(schema, exprs, inp)


@_translate_ir.register
def _(node: plrs._ir_nodes.Distinct, translator: Translator, schema: Schema) -> ir.IR:
    (keep, subset, maintain_order, zlice) = node.options
    keep = ir.Distinct._KEEP_MAP[keep]
    subset = frozenset(subset) if subset is not None else None
    return ir.Distinct(
        schema,
        keep,
        subset,
        zlice,
        maintain_order,
        translator.translate_ir(n=node.input),
    )


@_translate_ir.register
def _(node: plrs._ir_nodes.Sort, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        by = [
            translate_named_expr(translator, n=e, schema=inp.schema)
            for e in node.by_column
        ]
    stable, nulls_last, descending = node.sort_options
    order, null_order = sorting.sort_order(
        descending, nulls_last=nulls_last, num_keys=len(by)
    )
    # TODO: use the DynamicPred hint from node.slice (offset, length,
    # dynamic_pred_id). dynamic_pred_id is an int (u128 UUID) matching a
    # DynamicPred Function node in an upstream scan. The sort actor could
    # set a threshold predicate on it (e.g. col < max_seen) to prune rows
    # at the source.
    if not POLARS_VERSION_LT_142 and node.slice is not None:
        offset, length, *_ = node.slice
        zlice: Zlice | None = (offset, length)
    else:
        zlice = node.slice
    return ir.Sort(schema, by, order, null_order, stable, zlice, inp)


@_translate_ir.register
def _(node: plrs._ir_nodes.Slice, translator: Translator, schema: Schema) -> ir.IR:
    return ir.Slice(
        schema, node.offset, node.len, translator.translate_ir(n=node.input)
    )


@_translate_ir.register
def _(node: plrs._ir_nodes.Filter, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        mask = translate_predicate(
            translator,
            n=node.predicate.node,
            output_name=node.predicate.output_name,
            schema=inp.schema,
        )
        if mask is None:
            return inp
    return ir.Filter(schema, mask, inp)


@_translate_ir.register
def _(
    node: plrs._ir_nodes.SimpleProjection, translator: Translator, schema: Schema
) -> ir.IR:
    return ir.Projection(schema, translator.translate_ir(n=node.input))


@_translate_ir.register
def _(
    node: plrs._ir_nodes.MergeSorted, translator: Translator, schema: Schema
) -> ir.IR:
    key = node.key
    inp_left = translator.translate_ir(n=node.input_left)
    inp_right = translator.translate_ir(n=node.input_right)
    return ir.MergeSorted(
        schema,
        key,
        inp_left,
        inp_right,
    )


@_translate_ir.register
def _(
    node: plrs._ir_nodes.MapFunction, translator: Translator, schema: Schema
) -> ir.IR:
    name, *options = node.function
    return ir.MapFunction(
        schema,
        name,
        options,
        translator.translate_ir(n=node.input),
    )


@_translate_ir.register
def _(node: plrs._ir_nodes.Union, translator: Translator, schema: Schema) -> ir.IR:
    zlice = node.options if POLARS_VERSION_LT_141 else node.slice
    maintain_order = True if POLARS_VERSION_LT_141 else node.maintain_order
    return ir.Union(
        schema,
        zlice,
        maintain_order,
        *(translator.translate_ir(n=n) for n in node.inputs),
    )


@_translate_ir.register
def _(node: plrs._ir_nodes.HConcat, translator: Translator, schema: Schema) -> ir.IR:
    if POLARS_VERSION_LT_139:
        strict = False  # pragma: no cover
    else:
        _, strict, _ = node.options
    return ir.HConcat(
        schema,
        False,  # noqa: FBT003
        strict,
        *(translator.translate_ir(n=n) for n in node.inputs),
    )


@_translate_ir.register
def _(node: plrs._ir_nodes.Sink, translator: Translator, schema: Schema) -> ir.IR:
    payload = json.loads(node.payload)
    try:
        file = payload["File"]
        sink_kind_options = file[
            "file_type" if POLARS_VERSION_LT_136 else "file_format"
        ]
    except KeyError as err:  # pragma: no cover
        raise NotImplementedError("Unsupported payload structure") from err
    if isinstance(sink_kind_options, dict):
        if len(sink_kind_options) != 1:  # pragma: no cover; not sure if this can happen
            raise NotImplementedError("Sink options dict with more than one entry.")
        sink_kind, format_options = next(iter(sink_kind_options.items()))
    else:
        raise NotImplementedError(
            "Unsupported sink options structure"
        )  # pragma: no cover

    if POLARS_VERSION_LT_138:  # pragma: no cover
        sink_options = file.get("sink_options", {})
        cloud_options = file.get("cloud_options")
        options = format_options.copy()
        options.update(sink_options)
    else:
        unified_args = file.get("unified_sink_args", {})
        cloud_options = unified_args.get("cloud_options")
        options = {} if sink_kind == "NDJson" else format_options.copy()

        for k, v in unified_args.items():
            if k in {"mkdir", "maintain_order", "sync_on_close"}:
                options[k] = v

    if sink_kind in ("Csv", "NDJson", "Json"):
        compression = format_options.get("compression")
        if compression and compression != "Uncompressed":
            raise NotImplementedError(
                f"{sink_kind} compression ('{compression}') is not supported."
            )

    if POLARS_VERSION_LT_138:  # pragma: no cover
        path = file["target"]["Local"]
    else:
        path = file["target"]["inner"]

    return ir.Sink(
        schema=schema,
        kind=sink_kind,
        path=path,
        parquet_options=translator.config_options.parquet_options,
        options=options,
        cloud_options=cloud_options,
        df=translator.translate_ir(n=node.input),
    )


def translate_named_expr(
    translator: Translator, *, n: plrs._expr_nodes.PyExprIR, schema: Schema
) -> expr.NamedExpr:
    """
    Translate a polars-internal named expression IR object into our representation.

    Parameters
    ----------
    translator
        Translator object
    n
        Node to translate, a named expression node.
    schema
        Schema of the IR node this expression uses as evaluation context.

    Returns
    -------
    Translated IR object.

    Notes
    -----
    The datatype of the internal expression will be obtained from the
    visitor by calling ``get_dtype``, for this to work properly, the
    caller should arrange that the expression is translated with the
    node that it references "active" for the visitor (see :class:`set_node`).

    Raises
    ------
    NotImplementedError
        If any translation fails due to unsupported functionality.
    """
    return expr.NamedExpr(
        n.output_name, translator.translate_expr(n=n.node, schema=schema)
    )


def _contains_nested_lists(val: Any) -> bool:
    if not val:
        return False
    return any(isinstance(_, list) for _ in val)


@singledispatch
def _translate_expr(
    node: Any, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Function,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    name, *options = node.function_data
    options = tuple(options)
    if isinstance(name, plrs._expr_nodes.StringFunction):
        if name in {
            plrs._expr_nodes.StringFunction.StripChars,
            plrs._expr_nodes.StringFunction.StripCharsStart,
            plrs._expr_nodes.StringFunction.StripCharsEnd,
        }:
            column, chars = (
                translator.translate_expr(n=n, schema=schema) for n in node.input
            )
            if isinstance(chars, expr.Literal):
                # We check for null first because we want to use the
                # chars type, but it is invalid to try and
                # produce a string scalar with a null dtype.
                if chars.value is None:
                    # Polars uses None to mean "strip all whitespace"
                    chars = expr.Literal(column.dtype, "")
                elif chars.value == "":
                    # No-op in polars, but libcudf uses empty string
                    # as signifier to remove whitespace.
                    return column
            return expr.StringFunction(
                dtype,
                expr.StringFunction.Name.from_polars(name),
                options,
                column,
                chars,
            )
        return expr.StringFunction(
            dtype,
            expr.StringFunction.Name.from_polars(name),
            options,
            *(translator.translate_expr(n=n, schema=schema) for n in node.input),
        )
    elif isinstance(name, plrs._expr_nodes.BooleanFunction):
        if name == plrs._expr_nodes.BooleanFunction.IsBetween:
            column, lo, hi = (
                translator.translate_expr(n=n, schema=schema) for n in node.input
            )
            (closed,) = options
            lop, rop = expr.BooleanFunction._BETWEEN_OPS[closed]
            if not POLARS_VERSION_LT_139:
                # IsBetween type coercion does not
                # insert casts for Decimal/Float comparisons.
                column, lo, hi = _align_decimal_float_for_comparison(column, lo, hi)
            return expr.BinOp(
                dtype,
                plc.binaryop.BinaryOperator.LOGICAL_AND,
                expr.BinOp(dtype, lop, column, lo),
                expr.BinOp(dtype, rop, column, hi),
            )
        return expr.BooleanFunction(
            dtype,
            expr.BooleanFunction.Name.from_polars(name),
            options,
            *(translator.translate_expr(n=n, schema=schema) for n in node.input),
        )
    elif isinstance(name, plrs._expr_nodes.TemporalFunction):
        # functions for which evaluation of the expression may not return
        # the same dtype as polars, either due to libcudf returning a different
        # dtype, or due to our internal processing affecting what libcudf returns
        needs_cast = {
            plrs._expr_nodes.TemporalFunction.Year,
            plrs._expr_nodes.TemporalFunction.Month,
            plrs._expr_nodes.TemporalFunction.Day,
            plrs._expr_nodes.TemporalFunction.WeekDay,
            plrs._expr_nodes.TemporalFunction.Hour,
            plrs._expr_nodes.TemporalFunction.Minute,
            plrs._expr_nodes.TemporalFunction.Second,
            plrs._expr_nodes.TemporalFunction.Millisecond,
        }
        result_expr = expr.TemporalFunction(
            dtype,
            expr.TemporalFunction.Name.from_polars(name),
            options,
            *(translator.translate_expr(n=n, schema=schema) for n in node.input),
        )
        if name in needs_cast:
            return expr.Cast(dtype, True, result_expr)  # noqa: FBT003
        return result_expr
    elif isinstance(name, plrs._expr_nodes.StructFunction):
        return expr.StructFunction(
            dtype,
            expr.StructFunction.Name.from_polars(name),
            options,
            *(translator.translate_expr(n=n, schema=schema) for n in node.input),
        )
    elif _HAS_ROLLING_FUNCTION and isinstance(name, plrs._expr_nodes.RollingFunction):
        window_size, min_periods, weights, center, fn_params = options
        if weights is not None:
            raise NotImplementedError("Weighted rolling windows")
        RF = plrs._expr_nodes.RollingFunction
        agg_names = {
            RF.Sum: "sum",
            RF.Min: "min",
            RF.Max: "max",
            RF.Mean: "mean",
            RF.Var: "var",
            RF.Std: "std",
        }
        agg_name = agg_names.get(name)
        if agg_name is None:
            raise NotImplementedError(f"Unsupported rolling function: {name}")
        # Convert center + window_size to preceding/following for libcudf.
        # libcudf rolling_window semantics: element i uses elements
        # [i - preceding + 1, i + following].
        if center:
            following = (window_size - 1) // 2
            preceding = window_size - following
        else:
            preceding = window_size
            following = 0
        # Polars produces null when count <= ddof for var/std, but
        # libcudf produces NaN. Raise min_periods so that libcudf
        # returns null instead.
        if agg_name in ("var", "std"):
            (ddof,) = fn_params
            min_periods = max(min_periods, ddof + 1)
        (child,) = (translator.translate_expr(n=n, schema=schema) for n in node.input)
        return expr.FixedSizeRollingWindow(
            dtype, agg_name, preceding, following, min_periods, fn_params, child
        )
    elif isinstance(name, str):
        children = (translator.translate_expr(n=n, schema=schema) for n in node.input)
        if name == "rechunk":
            # Rechunking is a physical execution hint to Polars.
            # cudf-polars has no concept of chunking, so we can just
            # drop it.
            # Note: This could be a plan hook for explicit repartition for streaming engines
            # https://github.com/NVIDIA/cudf/pull/23192#discussion_r3553113408
            (child,) = children
            return child
        if name == "fused":
            # TODO: fuse into a single kernel via JIT transform, see
            # https://github.com/NVIDIA/cudf/issues/21456. We don't use
            # libcudf AST here because it widens the dtype for integer types
            # narrower than int32 (e.g. int8*int8 to int32), then fails
            # with a type mismatch when doing the add/sub with the third operand.
            (flavor,) = options
            a, b, c = children
            mul = plc.binaryop.BinaryOperator.MUL
            add = plc.binaryop.BinaryOperator.ADD
            sub = plc.binaryop.BinaryOperator.SUB
            match flavor:
                case "fma":
                    return expr.BinOp(dtype, add, expr.BinOp(dtype, mul, a, b), c)
                case "fsm":
                    return expr.BinOp(dtype, sub, a, expr.BinOp(dtype, mul, b, c))
                case "fms":
                    return expr.BinOp(dtype, sub, expr.BinOp(dtype, mul, a, b), c)
                case _:  # pragma: no cover
                    raise NotImplementedError(f"Unsupported fused expression {flavor=}")
        if name == "log" or (
            name == "l"
            and isinstance(options[0], str)
            and "".join((name, *options)) == "log"
        ):
            (child, base) = children
            assert isinstance(base, expr.Literal)
            return expr.BinOp(
                dtype,
                plc.binaryop.BinaryOperator.LOG_BASE,
                child,
                expr.Literal(dtype, base.value),
            )
        elif name == "pow":
            return expr.BinOp(dtype, plc.binaryop.BinaryOperator.POW, *children)
        elif name == "product":
            return expr.Agg(dtype, "product", None, translator._expr_context, *children)
        elif not POLARS_VERSION_LT_141 and name == "quantile":
            # polars >= 1.41 emits quantile as a string-named Function
            # expression (function_data=("quantile", interpolation)) with the
            # column and quantile value as inputs; earlier versions used a
            # dedicated Agg node.
            (interp,) = options
            return expr.Agg(
                dtype, "quantile", interp, translator._expr_context, *children
            )
        if name == "skew":
            (bias,) = options
            return expr.Skew(dtype, bias, *children)
        if name == "kurtosis":
            fisher, bias = options
            return expr.Kurtosis(dtype, fisher, bias, *children)
        if name == "arg_max" and len(options) == 2:
            # IRFunctionExpr::ArgSort is exposed as ("arg_max", descending, nulls_last)
            name = "arg_sort"
        return expr.UnaryFunction(dtype, name, options, *children)
    raise NotImplementedError(
        f"No handler for Expr function node with {name=}"
    )  # pragma: no cover; polars raises on the rust side for now


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Window,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    if hasattr(plrs._expr_nodes, "RollingGroupOptions") and isinstance(
        node.options, plrs._expr_nodes.RollingGroupOptions
    ):  # pragma: no cover; polars >=1.36 uses AExpr::Rolling now
        with set_expr_context(translator, ExecutionContext.ROLLING):
            agg = translator.translate_expr(n=node.function, schema=schema)
        name_generator = translator._internal_name_gen or unique_names(schema)
        aggs, named_post_agg = decompose_single_agg(
            expr.NamedExpr(next(name_generator), agg),
            name_generator,
            is_top=True,
            context=ExecutionContext.ROLLING,
        )
        named_aggs = [agg for agg, _ in aggs]
        orderby = node.options.index_column
        orderby_dtype = schema[orderby].plc_type
        if plc.traits.is_integral(orderby_dtype):
            # Integer orderby column is cast in implementation to int64 in polars
            orderby_dtype = plc.DataType(plc.TypeId.INT64)
        closed_window = node.options.closed_window
        if isinstance(named_post_agg.value, expr.Col):
            (named_agg,) = named_aggs
            return expr.RollingWindow(
                named_agg.value.dtype,
                orderby_dtype,
                node.options.offset,
                node.options.period,
                closed_window,
                orderby,
                named_agg.value,
            )
        replacements: dict[expr.Expr, expr.Expr] = {
            expr.Col(agg.value.dtype, agg.name): expr.RollingWindow(
                agg.value.dtype,
                orderby_dtype,
                node.options.offset,
                node.options.period,
                closed_window,
                orderby,
                agg.value,
            )
            for agg in named_aggs
        }
        return replace([named_post_agg.value], replacements)[0]
    elif isinstance(node.options, plrs._expr_nodes.WindowMapping):
        # pl.col("a").over(...)
        # Note: pl.col("a").rolling(...).over("g") also lands here. But
        # is not supported in polars < 1.39 since the AExpr::Rolling was
        # not exposed until polars 1.39.
        with set_expr_context(translator, ExecutionContext.WINDOW):
            agg = translator.translate_expr(n=node.function, schema=schema)
        name_gen = translator._internal_name_gen or unique_names(schema)
        aggs, post = decompose_single_agg(
            expr.NamedExpr(next(name_gen), agg),
            name_gen,
            is_top=True,
            context=ExecutionContext.WINDOW,
        )

        mapping = node.options.kind
        has_order_by = node.order_by is not None
        descending = bool(getattr(node, "order_by_descending", False))
        nulls_last = bool(getattr(node, "order_by_nulls_last", False))

        if mapping != "groups_to_rows":
            raise NotImplementedError(
                f"over(mapping_strategy) not supported yet: {mapping=}; "
                f"expected 'groups_to_rows'"
            )

        order_by_expr = (
            translator.translate_expr(n=node.order_by, schema=schema)
            if has_order_by
            else None
        )

        named_aggs = [agg for agg, _ in aggs]

        for named_agg in named_aggs:
            if has_order_by and isinstance(named_agg.value, expr.RollingWindow):
                raise NotImplementedError(
                    "rolling(...).over(..., order_by=...) is not supported"
                )
            if _unsupported_fill_over_window(named_agg.value):
                raise NotImplementedError(
                    "fill_null with strategy over a window is only supported when "
                    "applied directly to cum_sum()"
                )

        by_exprs = [
            translator.translate_expr(n=n, schema=schema) for n in node.partition_by
        ]

        child_deps: list[expr.Expr] = []
        for ne in named_aggs:
            v = ne.value
            if (
                isinstance(v, expr.UnaryFunction)
                and v.name == "fill_null_with_strategy"
                and isinstance(v.children[0], expr.UnaryFunction)
                and v.children[0].name == "cum_sum"
            ):
                child_deps.append(v.children[0].children[0])
            elif isinstance(v, expr.RollingWindow):
                child_deps.append(v.children[0])
                child_deps.append(expr.Col(schema[v.orderby], v.orderby))
            elif isinstance(v, (expr.FixedSizeRollingWindow, expr.Agg)) or (
                isinstance(v, expr.UnaryFunction)
                and v.name
                in {
                    "rank",
                    "fill_null_with_strategy",
                    "cum_sum",
                    "diff",
                    "shift",
                    "shift_and_fill",
                }
            ):
                child_deps.append(v.children[0])
        children = (*by_exprs, *((order_by_expr,) if has_order_by else ()), *child_deps)
        return expr.GroupedWindow(
            dtype,
            (mapping, has_order_by, descending, nulls_last),
            named_aggs,
            post,
            len(by_exprs),
            *children,
        )
    assert_never(node.options)


def _translate_rolling(
    node: plrs._expr_nodes.Rolling,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    # pl.col("a").rolling(...)
    with set_expr_context(translator, ExecutionContext.ROLLING):
        agg_expr = translator.translate_expr(n=node.function, schema=schema)
    orderby = translator.visitor.view_expression(node.index_column).name
    orderby_dtype = schema[orderby].plc_type
    if plc.traits.is_integral(orderby_dtype):
        orderby_dtype = plc.DataType(plc.TypeId.INT64)
    name_generator = unique_names(schema)
    aggs, named_post_agg = decompose_single_agg(
        expr.NamedExpr(next(name_generator), agg_expr),
        name_generator,
        is_top=True,
        context=ExecutionContext.ROLLING,
    )
    named_aggs = [a for a, _ in aggs]
    closed_window = node.closed_window
    if isinstance(named_post_agg.value, expr.Col):
        (named_agg,) = named_aggs
        return expr.RollingWindow(
            named_agg.value.dtype,
            orderby_dtype,
            node.offset,
            node.period,
            closed_window,
            orderby,
            named_agg.value,
        )
    replacements: dict[expr.Expr, expr.Expr] = {
        expr.Col(a.value.dtype, a.name): expr.RollingWindow(
            a.value.dtype,
            orderby_dtype,
            node.offset,
            node.period,
            closed_window,
            orderby,
            a.value,
        )
        for a in named_aggs
    }
    return replace([named_post_agg.value], replacements)[0]


# Rolling was added in polars 1.39. Use explicit registration (not annotation-based)
# so that Python 3.14's stricter get_type_hints() resolution does not fail when
# running on older polars versions that lack this node type.
if hasattr(plrs._expr_nodes, "Rolling"):
    _translate_expr.register(plrs._expr_nodes.Rolling)(_translate_rolling)


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Literal,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    if isinstance(node.value, plrs.PySeries):
        return expr.LiteralColumn(dtype, pl.Series._from_pyseries(node.value))
    if dtype.id() == plc.TypeId.LIST:  # pragma: no cover
        # TODO: Remove once pylibcudf.Scalar supports lists
        if _contains_nested_lists(node.value):
            return expr.LiteralColumn(
                dtype, pl.Series([node.value], dtype=dtype.polars_type)
            )
        return expr.LiteralColumn(dtype, pl.Series(node.value))
    if dtype.id() == plc.TypeId.STRUCT:
        raise NotImplementedError("Struct literals are not supported")
    return expr.Literal(dtype, node.value)


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Sort, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    # TODO: raise in groupby
    return expr.Sort(
        dtype, node.options, translator.translate_expr(n=node.expr, schema=schema)
    )


@_translate_expr.register
def _(
    node: plrs._expr_nodes.SortBy,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    options = node.sort_options
    return expr.SortBy(
        dtype,
        (options[0], tuple(options[1]), tuple(options[2])),
        translator.translate_expr(n=node.expr, schema=schema),
        *(translator.translate_expr(n=n, schema=schema) for n in node.by),
    )


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Slice,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    offset = translator.translate_expr(n=node.offset, schema=schema)
    length = translator.translate_expr(n=node.length, schema=schema)
    assert isinstance(offset, expr.Literal)
    assert isinstance(length, expr.Literal)
    return expr.Slice(
        dtype,
        offset.value,
        length.value,
        translator.translate_expr(n=node.input, schema=schema),
    )


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Gather,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    return expr.Gather(
        dtype,
        translator.translate_expr(n=node.expr, schema=schema),
        translator.translate_expr(n=node.idx, schema=schema),
    )


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Filter,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    return expr.Filter(
        dtype,
        translator.translate_expr(n=node.input, schema=schema),
        translator.translate_expr(n=node.by, schema=schema),
    )


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Cast, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    # TODO: node.options can be 2 meaning wrap_numerical=True
    # don't necessarily raise because wrapping isn't always needed, but it's unhandled
    strict = node.options != 1
    inner = translator.translate_expr(n=node.expr, schema=schema)

    if plc.traits.is_floating_point(inner.dtype.plc_type) and plc.traits.is_fixed_point(
        dtype.plc_type
    ):
        return expr.Cast(
            dtype,
            strict,
            expr.UnaryFunction(
                inner.dtype, "round", (-dtype.plc_type.scale(), "half_to_even"), inner
            ),
        )

    # Push casts into literals so we can handle Cast(Literal(Null))
    if isinstance(inner, expr.Literal):
        return inner.astype(dtype)
    else:
        return expr.Cast(dtype, strict, inner)


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Column,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    return expr.Col(dtype, node.name)


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Agg, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    agg_name = node.name
    args = [translator.translate_expr(n=arg, schema=schema) for arg in node.arguments]

    if (
        not POLARS_VERSION_LT_140
        and agg_name == "implode"
        and translator._expr_context
        in (ExecutionContext.GROUPBY, ExecutionContext.ROLLING)
    ):
        # polars >= 1.40 makes the per-group collect-to-list explicit: an
        # element-wise expression inside a groupby/rolling aggregation is now
        # wrapped in an explicit ``implode`` node. Earlier versions left it
        # implicit, emitting the bare expression (whose dtype already reflects
        # the per-group list). Unwrap the explicit ``implode`` so the existing
        # pointwise-collect path handles it as before.
        (child,) = args
        return child

    # libcudf does not support std/var on decimal types; cast to float first.
    if agg_name in {"std", "var"} and dtype.plc_type.id() in {
        plc.TypeId.FLOAT32,
        plc.TypeId.FLOAT64,
    }:
        args = [
            expr.Cast(dtype, True, arg)  # noqa: FBT003
            if plc.traits.is_fixed_point(arg.dtype.plc_type)
            else arg
            for arg in args
        ]

    value = expr.Agg(dtype, agg_name, node.options, translator._expr_context, *args)

    if agg_name in ("count", "n_unique") and value.dtype.id() != plc.TypeId.INT32:
        return expr.Cast(value.dtype, True, value)  # noqa: FBT003
    return value


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Ternary,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    return expr.Ternary(
        dtype,
        translator.translate_expr(n=node.predicate, schema=schema),
        translator.translate_expr(n=node.truthy, schema=schema),
        translator.translate_expr(n=node.falsy, schema=schema),
    )


@_translate_expr.register
def _(
    node: plrs._expr_nodes.BinaryExpr,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    left = translator.translate_expr(n=node.left, schema=schema)
    right = translator.translate_expr(n=node.right, schema=schema)
    if node.op == plrs._expr_nodes.Operator.TrueDivide and (
        plc.traits.is_fixed_point(left.dtype.plc_type)
        or plc.traits.is_fixed_point(right.dtype.plc_type)
    ):
        f64 = DataType(pl.Float64())
        return expr.Cast(
            dtype,
            True,  # noqa: FBT003
            expr.BinOp(
                f64,
                expr.BinOp._MAPPING[node.op],
                expr.Cast(f64, True, left),  # noqa: FBT003
                expr.Cast(f64, True, right),  # noqa: FBT003
            ),
        )

    if (
        node.op == plrs._expr_nodes.Operator.Multiply
        and plc.traits.is_fixed_point(left.dtype.plc_type)
        and plc.traits.is_fixed_point(right.dtype.plc_type)
    ):
        left_scale = -left.dtype.plc_type.scale()
        right_scale = -right.dtype.plc_type.scale()
        out_scale = max(left_scale, right_scale)

        return expr.UnaryFunction(
            DataType(pl.Decimal(38, out_scale)),
            "round",
            (out_scale, "half_to_even"),
            expr.BinOp(
                DataType(pl.Decimal(38, left_scale + right_scale)),
                expr.BinOp._MAPPING[node.op],
                left,
                right,
            ),
        )

    return expr.BinOp(
        dtype,
        expr.BinOp._MAPPING[node.op],
        left,
        right,
    )


@_translate_expr.register
def _(
    node: plrs._expr_nodes.Len, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    value = expr.Len(dtype)
    if dtype.id() != plc.TypeId.INT32:
        return expr.Cast(dtype, True, value)  # noqa: FBT003
    return value  # pragma: no cover; never reached since polars len has uint32 dtype
