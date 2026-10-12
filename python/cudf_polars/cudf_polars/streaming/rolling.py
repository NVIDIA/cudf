# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition Rolling lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import Rolling
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.dispatch import LowerIRTransformer


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
