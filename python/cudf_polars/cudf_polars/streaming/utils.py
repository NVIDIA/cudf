# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import operator
import warnings
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import (
    Col,
    FixedSizeRollingWindow,
    GroupedWindow,
    UnaryFunction,
)
from cudf_polars.dsl.ir import Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.base import PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def _concat(*dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return dfs[0] if len(dfs) == 1 else Union.do_evaluate(None, *dfs, context=context)


def _fallback_inform(
    msg: str, config_options: ConfigOptions[StreamingExecutor]
) -> None:
    """Inform the user of single-partition fallback."""
    match fallback_mode := config_options.executor.fallback_mode:
        case "warn":
            warnings.warn(msg, stacklevel=2)
        case "raise":
            raise NotImplementedError(msg)
        case "silent":
            pass
        case _:  # pragma: no cover; Should never get here.
            raise ValueError(
                f"{fallback_mode} is not a supported 'fallback_mode' "
                "option. Please use 'warn', 'raise', or 'silent'."
            )


def _dynamic_planning_on(config_options: ConfigOptions[StreamingExecutor]) -> bool:
    """Check if dynamic planning is enabled."""
    return config_options.executor.dynamic_planning is not None


def _lower_ir_fallback(
    ir: IR,
    rec: LowerIRTransformer,
    *,
    msg: str | None = None,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Catch-all single-partition lowering logic.
    # If any children contain multiple partitions,
    # those children will be collapsed with `Repartition`.
    from cudf_polars.streaming.repartition import Repartition
    from cudf_polars.streaming.select import _inline_hstack_false

    # Make sure we avoid mixed-length columns in intermediate TableChunks.
    ir = _inline_hstack_false(ir)

    if not ir.children:
        # nothing to lower or repartition.
        return ir, {ir: PartitionInfo(count=1)}

    # Lower children
    lowered_children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    inform = False
    for c in lowered_children:
        if partition_info[c].count > 1:
            inform = True
        child = Repartition(c.schema, c)
        partition_info[child] = PartitionInfo(count=1)
        children.append(child)

    if inform and msg:
        # Warn/raise the user if any children were collapsed
        # and the "fallback_mode" configuration is not "silent"
        _fallback_inform(msg, rec.state["config_options"])

    # Reconstruct and return
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info


def _leaf_column_names(expr: Expr) -> tuple[str, ...]:
    """Find the leaf column names of an expression."""
    if expr.children:
        return tuple(
            chain.from_iterable(_leaf_column_names(child) for child in expr.children)
        )
    elif isinstance(expr, Col):
        return (expr.name,)
    else:
        return ()


def _contains_over(exprs: Sequence[Expr]) -> bool:
    """Return True if any expression contains a window expression."""
    return any(isinstance(e, GroupedWindow) for e in traversal(exprs))


def _contains_unsupported_fill_strategy(exprs: Sequence[Expr]) -> bool:
    for e in traversal(exprs):
        if (
            isinstance(e, UnaryFunction)
            and e.name == "fill_null_with_strategy"
            and e.options[0] not in ("zero", "one")
        ):
            return True
    return False


_INPUT_ORDER_WINDOW_OPS = frozenset({"cum_sum", "diff", "shift", "shift_and_fill"})


def _contains_input_order_window_without_order_by(exprs: Sequence[Expr]) -> bool:
    """Return True for implicit input-order-sensitive window expressions."""
    for e in traversal(exprs):
        if not (isinstance(e, GroupedWindow) and not e.options[1]):
            continue
        for ne in e.named_aggs:
            v = ne.value
            if (
                isinstance(v, UnaryFunction)
                and v.name == "fill_null_with_strategy"
                and isinstance(v.children[0], UnaryFunction)
            ):
                v = v.children[0]
            if isinstance(v, UnaryFunction) and v.name in _INPUT_ORDER_WINDOW_OPS:
                return True
            if isinstance(v, FixedSizeRollingWindow):
                return True
    return False
