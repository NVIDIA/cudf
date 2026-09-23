# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import IR
from cudf_polars.dsl.tracing import log_do_evaluate
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.utils import _dynamic_planning_on

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.dispatch import LowerIRTransformer
    from cudf_polars.streaming.parallel import PartitionInfo
    from cudf_polars.typing import Schema


class Shuffle(IR):
    """
    Shuffle multi-partition data.

    Notes
    -----
    Only hash-based partitioning is supported (for now).
    """

    __slots__ = ("keys",)
    _non_child = (
        "schema",
        "keys",
    )
    _n_non_child_args = 2
    keys: tuple[NamedExpr, ...]
    """Keys to shuffle on."""

    def __init__(
        self,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        df: IR,
    ):
        self.schema = schema
        self.keys = keys
        self._non_child_args = (schema, keys)
        self.children = (df,)

    # the type-ignore is for
    # Argument 1 to "log_do_evaluate" has incompatible type "Callable[[type[Shuffle], <snip>]"
    #    expected Callable[[type[IR], <snip>]
    # But Shuffle is a subclass of IR, so this is fine.
    @classmethod
    @log_do_evaluate  # type: ignore[arg-type]  # (decorator accepts IR subclass evaluator)
    def do_evaluate(
        cls,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition Shuffle evaluation is a no-op
        return df


@lower_ir_node.register(Shuffle)
def _(
    ir: Shuffle, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Simple lower_ir_node handling for the default hash-based shuffle.
    # More-complex logic (e.g. joining and sorting) should
    # be handled separately.
    from cudf_polars.streaming.parallel import PartitionInfo

    (child,) = ir.children

    # Check for dynamic planning - may have more partitions at runtime
    config_options = rec.state["config_options"]
    new_child, pi = rec(child)
    already_partitioned = ir.keys == pi[new_child].partitioned_on
    single_partition = pi[new_child].count == 1 and not _dynamic_planning_on(
        config_options
    )
    if single_partition or already_partitioned:
        # Already shuffled
        return new_child, pi
    new_node = ir.reconstruct([new_child])
    pi[new_node] = PartitionInfo(
        # Default shuffle preserves partition count
        count=pi[new_child].count,
        # Add partitioned_on info
        partitioned_on=ir.keys,
    )
    return new_node, pi
