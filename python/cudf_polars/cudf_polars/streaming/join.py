# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Join Logic."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import ConditionalJoin, Join, Slice
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.shuffle import Shuffle
from cudf_polars.streaming.utils import (
    _dynamic_planning_on,
    _fallback_inform,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.parallel import LowerIRTransformer


def _maybe_shuffle_frame(
    frame: IR,
    on: tuple[NamedExpr, ...],
    partition_info: MutableMapping[IR, PartitionInfo],
    output_count: int,
) -> IR:
    # Shuffle `frame` if it isn't already shuffled.
    if (
        partition_info[frame].partitioned_on == on
        and partition_info[frame].count == output_count
    ):
        # Already shuffled
        return frame
    else:
        # Insert new Shuffle node
        frame = Shuffle(
            frame.schema,
            on,
            frame,
        )
        partition_info[frame] = PartitionInfo(
            count=output_count,
            partitioned_on=on,
        )
        return frame


def _make_hash_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Shuffle left and right dataframes (if necessary)
    left = _maybe_shuffle_frame(
        left,
        ir.left_on,
        partition_info,
        output_count,
    )
    right = _maybe_shuffle_frame(
        right,
        ir.right_on,
        partition_info,
        output_count,
    )
    # Reconstruct with the lowered and possibly shuffled children.
    ir = ir.reconstruct([left, right])

    # Record new partitioning info
    partitioned_on: tuple[NamedExpr, ...] = ()
    if ir.left_on == ir.right_on or (ir.options[0] in ("Left", "Semi", "Anti")):
        partitioned_on = ir.left_on
    elif ir.options[0] == "Right":
        partitioned_on = ir.right_on
    partition_info[ir] = PartitionInfo(
        count=output_count,
        partitioned_on=partitioned_on,
    )

    return ir, partition_info


def _should_bcast_join(
    ir: Join,
    left: IR,
    right: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    output_count: int,
    broadcast_limit: int,
    target_partition_size: int,
) -> bool:
    # Decide if a broadcast join is appropriate.
    if partition_info[left].count >= partition_info[right].count:
        small_count = partition_info[right].count
        large = left
        large_on = ir.left_on
    else:
        small_count = partition_info[left].count
        large = right
        large_on = ir.right_on

    # Avoid the broadcast if the "large" table is already shuffled
    large_shuffled = (
        partition_info[large].partitioned_on == large_on
        and partition_info[large].count == output_count
    )

    # Derive a partition-count threshold: how many target-sized partitions fit
    # in the broadcast byte budget?
    bcast_partition_threshold = broadcast_limit // target_partition_size

    return (
        not large_shuffled
        and small_count <= bcast_partition_threshold
        and (
            ir.options[0] == "Inner"
            or (ir.options[0] in ("Left", "Semi", "Anti") and large == left)
            or (ir.options[0] == "Right" and large == right)
        )
    )


def _make_bcast_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    new_node = ir.reconstruct([left, right])
    partition_info[new_node] = PartitionInfo(count=output_count)
    return new_node, partition_info


def _has_non_pointwise_keys(ir: Join) -> bool:
    keys = [ne.value for keys in (ir.left_on, ir.right_on) for ne in keys]
    return not all(expr.is_pointwise for expr in traversal(keys))


@lower_ir_node.register(ConditionalJoin)
def _(
    ir: ConditionalJoin, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if ir.options[2]:  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Slice not supported in ConditionalJoin for multiple partitions.",
        )

    config_options = rec.state["config_options"]
    dynamic_planning = _dynamic_planning_on(config_options)

    # Lower children
    left, right = ir.children
    left, pi_left = rec(left)
    right, pi_right = rec(right)

    # Fallback to single partition on the smaller table whenever either
    # side has more than one partition.
    left_count = pi_left[left].count
    right_count = pi_right[right].count
    output_count = max(left_count, right_count)
    if output_count > 1 or dynamic_planning:
        if left_count < right_count:
            left = Repartition(left.schema, left)
            pi_left[left] = PartitionInfo(count=1)
        else:
            right = Repartition(right.schema, right)
            pi_right[right] = PartitionInfo(count=1)
        _fallback_inform(
            "ConditionalJoin not supported for multiple partitions.",
            config_options,
        )

    # Reconstruct and return
    new_node = ir.reconstruct([left, right])
    partition_info = reduce(operator.or_, (pi_left, pi_right))
    partition_info[new_node] = PartitionInfo(count=output_count)
    return new_node, partition_info


@lower_ir_node.register(Join)
def _(
    ir: Join, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Pull slice operations out of the Join before lowering
    if (zlice := ir.options[2]) is not None:
        offset, length = zlice
        if length is None:  # pragma: no cover
            return _lower_ir_fallback(
                ir,
                rec,
                msg="This slice not supported for multiple partitions.",
            )
        new_join = Join(
            ir.schema,
            ir.left_on,
            ir.right_on,
            (*ir.options[:2], None, *ir.options[3:]),
            *ir.children,
        )
        return rec(Slice(ir.schema, offset, length, new_join))

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Check for dynamic planning - may have more partitions at runtime
    config_options = rec.state["config_options"]
    dynamic_planning = _dynamic_planning_on(config_options)

    left, right = children
    output_count = max(partition_info[left].count, partition_info[right].count)
    has_non_pointwise_keys = _has_non_pointwise_keys(ir)
    if output_count == 1 and not dynamic_planning:
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info
    elif ir.options[0] == "Cross":  # pragma: no cover
        return _lower_ir_fallback(
            ir, rec, msg="Cross join not support for multiple partitions."
        )

    maintain_order = ir.options[5]
    if maintain_order != "none" and (output_count > 1 or dynamic_planning):
        return _lower_ir_fallback(
            ir,
            rec,
            msg=f"Join({maintain_order=}) not supported for multiple partitions.",
        )

    # Check for dynamic planning - defer broadcast vs shuffle decision to runtime
    if dynamic_planning:  # pragma: no cover; Requires rapidsmpf runtime
        if has_non_pointwise_keys:
            return _lower_ir_fallback(
                ir,
                rec,
                msg="Multi-partition Join not supported for non-pointwise key expressions.",
            )
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo(count=output_count)
        return new_node, partition_info

    if _should_bcast_join(
        ir,
        left,
        right,
        partition_info,
        output_count,
        config_options.executor.broadcast_limit,
        config_options.executor.target_partition_size,
    ):
        # Create a broadcast join
        return _make_bcast_join(
            ir,
            output_count,
            partition_info,
            left,
            right,
        )
    elif has_non_pointwise_keys:
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Multi-partition Join not supported for non-pointwise key expressions.",
        )
    else:
        # Create a hash join
        return _make_hash_join(
            ir,
            output_count,
            partition_info,
            left,
            right,
        )
