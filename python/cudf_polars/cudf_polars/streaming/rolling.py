# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition Rolling lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from cudf_polars.dsl.ir import IR, Rolling, Select
from cudf_polars.dsl.tracing import log_do_evaluate, nvtx_annotate_cudf_polars
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl import expr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema


class FixedSizeRolling(IR):
    """Evaluate fixed-size rolling expressions with row-count overlap."""

    __slots__ = ("exprs", "following_overlap", "preceding_overlap", "should_broadcast")
    _non_child: ClassVar[tuple[str, ...]] = (
        "schema",
        "exprs",
        "should_broadcast",
        "preceding_overlap",
        "following_overlap",
    )
    _n_non_child_args: ClassVar[int] = 2
    _preserves_output_order: ClassVar[bool] = True
    exprs: tuple[expr.NamedExpr, ...]
    should_broadcast: bool
    preceding_overlap: int
    following_overlap: int

    def __init__(
        self,
        schema: Schema,
        exprs: Sequence[expr.NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        preceding_overlap: int,
        following_overlap: int,
        df: IR,
    ) -> None:
        self.schema = schema
        self.exprs = tuple(exprs)
        self.should_broadcast = should_broadcast
        self.preceding_overlap = preceding_overlap
        self.following_overlap = following_overlap
        self.children = (df,)
        self._non_child_args = (self.exprs, should_broadcast)

    @classmethod
    @log_do_evaluate
    @nvtx_annotate_cudf_polars(message="FixedSizeRolling")
    def do_evaluate(
        cls,
        exprs: tuple[expr.NamedExpr, ...],
        should_broadcast: bool,  # noqa: FBT001
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate fixed-size rolling expressions against df."""
        return Select.do_evaluate(exprs, should_broadcast, df, context=context)


@lower_ir_node.register(FixedSizeRolling)
def _(
    ir: FixedSizeRolling, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Lower fixed-size rolling expressions for streaming execution."""
    (child,) = ir.children
    child, partition_info = rec(child)
    new_node = ir.reconstruct([child])
    partition_info[new_node] = PartitionInfo(count=partition_info[child].count)
    return new_node, partition_info


@lower_ir_node.register(Rolling)
def _(
    ir: Rolling, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Lower Rolling for streaming execution."""
    if len(ir.keys) > 0 or ir.zlice is not None:
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Grouped or sliced rolling does not support multiple partitions.",
        )
    (child,) = ir.children
    child, partition_info = rec(child)
    new_node = ir.reconstruct([child])
    partition_info[new_node] = PartitionInfo(count=partition_info[child].count)
    return new_node, partition_info
