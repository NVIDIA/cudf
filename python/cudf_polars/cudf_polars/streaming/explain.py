# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explain logical and physical plans."""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import functools
import os
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.expressions.base import Col, ColRef
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.expressions.ternary import Ternary
from cudf_polars.dsl.expressions.unary import Cast, UnaryFunction
from cudf_polars.dsl.ir import (
    ConditionalJoin,
    Filter,
    GroupBy,
    Join,
    PythonScan,
    Scan,
    Sort,
)
from cudf_polars.dsl.translate import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.base import IOPartitionFlavor
from cudf_polars.streaming.filter_hint import JoinWithPrefilter, PushdownFilterHint
from cudf_polars.streaming.io import StreamingScan, scan_partition_plan
from cudf_polars.streaming.parallel import lower_ir_graph, optimize_with_stats
from cudf_polars.streaming.statistics import (
    collect_statistics,
)
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import polars as pl

    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import PartitionInfo, StatsCollector


@dataclasses.dataclass
class PartitionPlanRow:
    """One row of the partition plan summary table."""

    query: int
    table: str
    flavor: IOPartitionFlavor
    factor: int
    files: int
    projected_bytes: int
    task_bytes: int
    partitions: int


def explain_query(
    q: pl.LazyFrame,
    engine: pl.GPUEngine,
    *,
    optimized: bool = True,
    physical: bool = True,
    executor: concurrent.futures.Executor | None = None,
) -> str:
    """
    Return a formatted string representation of the IR plan.

    Parameters
    ----------
    q : pl.LazyFrame
        The LazyFrame to explain.
    engine : pl.GPUEngine
        The configured GPU engine to use.
    optimized
        If True and showing the logical plan, run cudf-polars specific
        query optimization.
    physical : bool, default True
        If True, show the physical (lowered) plan.
        If False, show the logical (pre-lowering) plan.
    executor
        Optional executor to use for IO operations. This function does not start
        or shutdown the executor. If not provided, a new thread pool executor
        is created and used.

    Returns
    -------
    str
        A string representation of the IR plan.
    """
    cm: contextlib.AbstractContextManager[concurrent.futures.Executor]

    if executor is None:
        cm = executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="cudf-polars-explain"
        )
    else:
        # we only shut down the executor if we created it.
        cm = contextlib.nullcontext(executor)

    config = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()

    if physical:
        with cm:
            stats = collect_statistics(ir, config, executor)
        lowered = lower_ir_graph(ir, config, stats)
        return _repr_ir_tree(
            lowered.lowered, lowered.partition_info, stats=stats, config=config
        )
    else:
        if config.executor.name == "streaming":
            # Include row-count statistics for the logical plan
            with cm:
                stats = collect_statistics(ir, config, executor)
            if optimized:
                ir = optimize_with_stats(ir, config, stats)
            return _repr_ir_tree(ir, stats=stats)
        else:
            return _repr_ir_tree(ir)


def collect_partition_plan(
    q: pl.LazyFrame,
    engine: pl.GPUEngine,
    q_id: int,
) -> list[PartitionPlanRow]:
    """
    Return one PartitionPlanRow per unique StreamingScan in the physical plan.

    Deduplicates scans that appear multiple times due to subquery structure.
    """
    config = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()

    with concurrent.futures.ThreadPoolExecutor(
        thread_name_prefix="cudf-polars-explain"
    ) as executor:
        stats = collect_statistics(ir, config, executor)
    lowered = lower_ir_graph(ir, config, stats)
    lowered_ir = lowered.lowered
    partition_info = lowered.partition_info

    seen: set[tuple] = set()
    rows: list[PartitionPlanRow] = []

    for node in traversal([lowered_ir]):
        if not isinstance(node, StreamingScan):
            continue
        base_scan = node.base_scan

        dedup_key = (tuple(base_scan.paths), tuple(sorted(base_scan.schema.keys())))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        source = stats.scan_stats.get(base_scan)
        if source is None:
            continue

        plan = scan_partition_plan(base_scan, stats, config)
        projected_bytes = sum(
            sz
            for col in base_scan.schema
            if (sz := source.column_storage_size(col)) is not None
        )
        partitions = partition_info[node].count
        factor = plan.factor
        flavor = plan.flavor

        match flavor:
            case IOPartitionFlavor.SPLIT_FILES:
                files = partitions // factor if factor > 0 else partitions
                task_bytes = (
                    projected_bytes // factor if factor > 0 else projected_bytes
                )
            case IOPartitionFlavor.FUSED_FILES:
                files = partitions * factor
                task_bytes = projected_bytes * factor
            case _:
                files = partitions
                task_bytes = projected_bytes

        p = Path(base_scan.paths[0])
        stem = p.stem
        parent = p.parent.name
        # Prefer the stem unless it looks like a partition filename (purely
        # numeric like "1" or prefixed like "part-0"), in which case the
        # parent directory holds the table name.
        table = parent if (stem.isdigit() or stem.lower().startswith("part")) else stem

        rows.append(
            PartitionPlanRow(
                query=q_id,
                table=table,
                flavor=flavor,
                factor=factor,
                files=files,
                projected_bytes=projected_bytes,
                task_bytes=task_bytes,
                partitions=partitions,
            )
        )

    return rows


def _fmt_partition_bytes(b: int) -> str:
    if b < 1_000:
        return f"{b} B"
    elif b < 1_000_000:
        return f"{round(b / 1_000, 2):g} KB"
    elif b < 1_000_000_000:
        return f"{round(b / 1_000_000, 2):g} MB"
    else:
        return f"{round(b / 1_000_000_000, 2):g} GB"


def factor_str(row: PartitionPlanRow) -> str:
    """Format the factor field with units appropriate to the scan flavor."""
    match row.flavor:
        case IOPartitionFlavor.SPLIT_FILES:
            return f"{row.factor} tasks/file"
        case IOPartitionFlavor.FUSED_FILES:
            unit = "file" if row.factor == 1 else "files"
            return f"{row.factor} {unit}/task"
        case _:
            return str(row.factor)


def format_partition_plan_table(rows: list[PartitionPlanRow]) -> str:
    """Format a list of PartitionPlanRows as a fixed-width ASCII table."""
    if not rows:
        return ""

    headers = [
        "Q",
        "Table",
        "Flavor",
        "Factor",
        "Files",
        "Projected (bytes/file)",
        "Size/task",
        "Partitions",
    ]

    formatted: list[list[str]] = []
    prev_q: int | None = None
    for row in rows:
        q_str = str(row.query) if row.query != prev_q else ""
        prev_q = row.query
        formatted.append(
            [
                q_str,
                row.table,
                row.flavor.name,
                factor_str(row),
                str(row.files),
                _fmt_partition_bytes(row.projected_bytes),
                _fmt_partition_bytes(row.task_bytes),
                str(row.partitions),
            ]
        )

    col_widths = [len(h) for h in headers]
    for cells in formatted:
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )
    lines = ["", "Partition Plan Summary", sep, header_row, sep]
    lines.extend(
        "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"
        for cells in formatted
    )
    lines.append(sep)
    return "\n".join(lines)


def _fmt_row_count(value: int | None) -> str:
    """Format a row count as a readable string."""
    if value is None:
        return ""
    elif value < 1_000:
        return f"{value}"
    elif value < 1_000_000:
        return f"{round(value / 1_000, 2):g} K"
    elif value < 1_000_000_000:
        return f"{round(value / 1_000_000, 2):g} M"
    else:
        return f"{round(value / 1_000_000_000, 2):g} B"


def _repr_ir_tree(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
    stats: StatsCollector | None = None,
    config: ConfigOptions | None = None,
) -> str:
    header = _repr_ir(ir, offset=offset)
    count = partition_info[ir].count if partition_info else None
    if stats is not None and (source := stats.scan_stats.get(ir)) is not None:
        # Only annotate leaf scan nodes that have a row-count estimate
        row_count_estimate = _fmt_row_count(source.row_count)
        row_count = f"~{row_count_estimate}" if row_count_estimate else "unknown"
        header = header.rstrip("\n") + f" {row_count=}\n"
    if (
        os.environ.get("CUDF_POLARS__EXPLAIN__PARTITION_PLAN", "0") == "1"
        and config is not None
        and stats is not None
        and isinstance(ir, StreamingScan)
        and (source := stats.scan_stats.get(ir.base_scan)) is not None
    ):
        plan = scan_partition_plan(ir.base_scan, stats, config)
        projected_size = sum(
            sz
            for col in ir.base_scan.schema
            if (sz := source.column_storage_size(col)) is not None
        )
        plan_info = (
            f"flavor={plan.flavor.name} factor={plan.factor}"
            f" projected={_fmt_partition_bytes(projected_size)}"
        )
        header = header.rstrip("\n") + f" [{plan_info}]\n"
    if isinstance(ir, PythonScan):
        # The lowered partition count is a placeholder. under dynamic planning the
        # runtime adapts to the real chunk count, so don't report a misleading value.
        header = header.rstrip("\n") + " [unknown]\n"
    elif count is not None:
        header = header.rstrip("\n") + f" [{count}]\n"

    children_strs = [
        _repr_ir_tree(
            child, partition_info, offset=offset + "  ", stats=stats, config=config
        )
        for child in ir.children
    ]

    return header + "".join(
        f"{line}{offset}  (repeated {count} times)\n"
        if (count := sum(1 for _ in group)) > 1
        else line
        for line, group in groupby(children_strs)
    )


def _repr_schema(schema: tuple | None) -> str:
    if schema is None:
        return ""  # pragma: no cover; no test yet
    names = tuple(schema)
    if len(names) > 6:
        names = (*names[:3], "...", *names[-2:])
    return f" {names}"


def _repr_header(offset: str, label: str, schema: tuple | dict | None) -> str:
    return f"{offset}{label}{_repr_schema(tuple(schema) if schema is not None else None)}\n"


@functools.singledispatch
def _repr_ir(ir: IR, *, offset: str = "") -> str:
    return _repr_header(offset, type(ir).__name__.upper(), ir.schema)


@_repr_ir.register
def _(ir: GroupBy, *, offset: str = "") -> str:
    keys = tuple(ne.name for ne in ir.keys)
    return _repr_header(offset, f"GROUPBY {keys}", ir.schema)


@_repr_ir.register
def _(ir: Join, *, offset: str = "") -> str:
    left_on = tuple(ne.name for ne in ir.left_on)
    right_on = tuple(ne.name for ne in ir.right_on)
    return _repr_header(offset, f"JOIN {ir.options[0]} {left_on} {right_on}", ir.schema)


@_repr_ir.register
def _(ir: JoinWithPrefilter, *, offset: str = "") -> str:
    left_on = tuple(ne.name for ne in ir.left_on)
    right_on = tuple(ne.name for ne in ir.right_on)
    prefilters = tuple(type(prefilter.domain).__name__ for prefilter in ir.prefilters)
    return _repr_header(
        offset,
        f"JOIN {ir.options[0]} {left_on} {right_on} {prefilters=}",
        ir.schema,
    )


@_repr_ir.register
def _(ir: PushdownFilterHint, *, offset: str = "") -> str:
    target_on = tuple(ne.name for ne in ir.target_on)
    domain_on = tuple(ne.name for ne in ir.domain_on)
    return _repr_header(
        offset,
        f"PUSHDOWN FILTER HINT {target_on} {domain_on} {ir.placement}",
        ir.schema,
    )


_BinaryOperator = plc.binaryop.BinaryOperator
_BINOP_SYMBOLS: dict[_BinaryOperator, str] = {
    _BinaryOperator.EQUAL: "==",
    _BinaryOperator.NOT_EQUAL: "!=",
    _BinaryOperator.LESS: "<",
    _BinaryOperator.LESS_EQUAL: "<=",
    _BinaryOperator.GREATER: ">",
    _BinaryOperator.GREATER_EQUAL: ">=",
    _BinaryOperator.LOGICAL_AND: "&",
    _BinaryOperator.NULL_LOGICAL_AND: "&",
    _BinaryOperator.LOGICAL_OR: "|",
    _BinaryOperator.NULL_LOGICAL_OR: "|",
}


def _predicate_to_str(expr: Expr) -> str:
    match expr:
        case Col(name=name):
            return name
        case ColRef():
            col = expr.children[0]
            assert isinstance(col, Col)
            return col.name
        case Literal(value=value):
            return repr(value)
        case Cast():
            return _predicate_to_str(expr.children[0])
        case BinOp(op=op):
            left, right = expr.children
            sym = _BINOP_SYMBOLS.get(op, op.name)
            return f"({_predicate_to_str(left)} {sym} {_predicate_to_str(right)})"
        case UnaryFunction(name=name):
            # Unlike the other cases here, UnaryFunction doesn't have a fixed
            # number of children. E.g. `pl.col("x").fill_null(0)` has two:
            # the column expression ("x") and the fill value literal (0).
            args = ", ".join(_predicate_to_str(child) for child in expr.children)
            return f"{name}({args})"
        case Ternary():
            when, then, otherwise = expr.children
            return f"when({_predicate_to_str(when)}).then({_predicate_to_str(then)}).otherwise({_predicate_to_str(otherwise)})"
        case _:
            return type(expr).__name__


@_repr_ir.register
def _(ir: ConditionalJoin, *, offset: str = "") -> str:
    pred = _predicate_to_str(ir.predicate)
    return _repr_header(offset, f"CONDITIONALJOIN {pred}", ir.schema)


@_repr_ir.register
def _(ir: Filter, *, offset: str = "") -> str:
    pred = _predicate_to_str(ir.mask.value)
    return _repr_header(offset, f"FILTER {pred}", ir.schema)


@_repr_ir.register
def _(ir: Sort, *, offset: str = "") -> str:
    by = tuple(ne.name for ne in ir.by)
    return _repr_header(offset, f"SORT {by}", ir.schema)


@_repr_ir.register
def _(ir: Scan, *, offset: str = "") -> str:
    label = f"SCAN {ir.typ.upper()}"
    if ir.predicate is not None:
        label += f" {_predicate_to_str(ir.predicate.value)}"
    return _repr_header(offset, label, ir.schema)
