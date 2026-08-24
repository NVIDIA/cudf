# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition Distinct logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.expressions.base import Col, NamedExpr
from cudf_polars.dsl.ir import Distinct
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.shuffle import Shuffle
from cudf_polars.streaming.utils import (
    _dynamic_planning_on,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def _distinct_keys(ir: Distinct) -> tuple[NamedExpr, ...]:
    subset: frozenset[str] = ir.subset or frozenset(ir.schema)
    return tuple(
        NamedExpr(name, Col(ir.schema[name], name))
        for name in ir.schema
        if name in subset
    )


def _lower_distinct_static(
    ir: Distinct,
    child: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Lower a Distinct IR into partition-wise stages with static planning.

    Note: Edge cases (KEEP_NONE + ordering, complex slice, pre-shuffle)
    must be handled by the caller before calling this function.

    Parameters
    ----------
    ir
        The Distinct IR node to lower.
    child
        The reconstructed child of ``ir``. May differ from ``ir.children[0]``.
        If KEEP_NONE, caller must ensure child is already shuffled on keys.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.

    Returns
    -------
    new_node
        The lowered Distinct node.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    child_count = partition_info[child].count

    n_ary = 32
    if ir.zlice is not None and ir.zlice[1] is not None:
        n_ary = max(1_000_000 // ir.zlice[1], 2)

    # Partition-wise unique
    count = child_count
    new_node: IR = ir.reconstruct([child])
    partition_info[new_node] = PartitionInfo(count=count)

    # Tree reduction
    while count > 1:
        new_node = Repartition(new_node.schema, new_node)
        count = max(math.ceil(count / n_ary), 1)
        partition_info[new_node] = PartitionInfo(count=count)
        new_node = ir.reconstruct([new_node])
        partition_info[new_node] = PartitionInfo(count=count)

    return new_node, partition_info


def lower_distinct(
    ir: Distinct,
    child: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Lower a Distinct IR with an already-lowered child.

    Note: Edge cases (KEEP_NONE + ordering, complex slice, pre-shuffle)
    must be handled by the caller before calling this function.
    """
    if _dynamic_planning_on(
        config_options
    ):  # pragma: no cover; Requires rapidsmpf runtime
        # Dynamic planning: Reconstruct the Distinct.
        # The runtime distinct_node will handle strategy selection,
        # respecting ir.stable, ir.keep, and ir.zlice attributes.
        dynamic_node = ir.reconstruct([child])
        partition_info[dynamic_node] = PartitionInfo(
            count=partition_info[child].count,
            partitioned_on=_distinct_keys(ir),
        )
        return dynamic_node, partition_info

    return _lower_distinct_static(ir, child, partition_info)


@lower_ir_node.register(Distinct)
def _(
    ir: Distinct, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Extract child partitioning
    child, partition_info = rec(ir.children[0])
    child_count = partition_info[child].count
    distinct_keys = _distinct_keys(ir)

    config_options = rec.state["config_options"]

    # Check for ordering requirements (shuffle is not stable)
    require_tree = ir.stable or ir.keep in (
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
    )

    # Handle edge cases upfront
    # KEEP_NONE + ordering = unsupported (need shuffle but shuffle is unstable)
    if ir.keep == plc.stream_compaction.DuplicateKeepOption.KEEP_NONE:
        if require_tree:
            return _lower_ir_fallback(
                ir,
                rec,
                msg="Unsupported unique options for multiple partitions.",
            )

        # KEEP_NONE needs pre-shuffle (must see all duplicates to drop them)
        if partition_info[child].partitioned_on != distinct_keys:
            child = Shuffle(
                child.schema,
                distinct_keys,
                child,
            )
            partition_info[child] = PartitionInfo(
                count=child_count,
                partitioned_on=distinct_keys,
            )

    # Complex slice = unsupported (offset >= 1 or no length)
    if ir.zlice is not None and (ir.zlice[0] >= 1 or ir.zlice[1] is None):
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Complex slice not supported for multiple partitions.",
        )

    return lower_distinct(
        ir,
        child,
        partition_info,
        config_options,
    )
