# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sorting Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Slice, Sort
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.utils import (
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import PartitionInfo
    from cudf_polars.streaming.dispatch import LowerIRTransformer


def _has_simple_zlice(zlice: tuple[int, int | None] | None) -> bool:
    """Check if a zlice is a simple top-k/bottom-k operation."""
    if zlice is None:
        return False
    has_offset = zlice[0] > 0 or (
        zlice[0] < 0 and zlice[1] is not None and zlice[0] + zlice[1] < 0
    )
    return not has_offset


@lower_ir_node.register(Sort)
def _(
    ir: Sort, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check sort keys
    if not all(
        isinstance(expr, Col) for expr in traversal([e.value for e in ir.by])
    ):  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="sort currently only supports column names as `by` keys.",
        )

    if ir.zlice is not None and not _has_simple_zlice(ir.zlice):
        # Pull "complex" slices out of the Sort node altogether.
        return rec(
            Slice(
                ir.schema,
                *ir.zlice,
                Sort(
                    ir.schema,
                    ir.by,
                    ir.order,
                    ir.null_order,
                    ir.stable,
                    None,
                    ir.children[0],
                ),
            )
        )

    # Extract child partitioning
    child, partition_info = rec(ir.children[0])

    sort_node = ir.reconstruct([child])
    partition_info[sort_node] = partition_info[child]
    return sort_node, partition_info
