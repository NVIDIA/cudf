# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Over IR node for streaming window expressions."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, cast

from cudf_polars.dsl.expr import Agg, Col, Len, NamedExpr
from cudf_polars.dsl.ir import IR, GroupBy, Select
from cudf_polars.dsl.utils.naming import names_to_indices, unique_names
from cudf_polars.streaming.groupby import combine, decompose

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import GroupedWindow
    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.base import PartitionInfo
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions


# Aggregations whose partial results can be combined.
_DECOMPOSABLE_AGG_NAMES: frozenset[str] = frozenset(
    ("sum", "count", "mean", "min", "max", "std", "var")
)


def _build_over_groupby_irs(
    gw_nodes: tuple[GroupedWindow, ...],
    child_ir: IR,
) -> tuple[GroupBy, GroupBy, Select]:
    """
    Build piecewise, reduction, and selection GroupBy IRs.

    Parameters
    ----------
    gw_nodes
        Top-level GroupedWindow nodes sharing the same partition-by keys;
        all must be scalar (Agg/Len only in named_aggs).
    child_ir
        Input IR feeding the Over node; defines the schema seen by the
        per-chunk piecewise GroupBy.

    Returns
    -------
    piecewise_ir
        GroupBy IR that computes partial aggregates per chunk.
    reduction_ir
        GroupBy IR that reduces partial aggregates to a single result.
    agg_select_ir
        Select IR applied on top of the reduction. Carries any post-
        aggregation expressions (e.g. division for mean); for fully
        pass-through aggregations it is a Select of plain ``Col`` refs
        of the same shape as the reduction output.
    """
    gw = gw_nodes[0]
    by_exprs = cast("list[Col]", list(gw.children[: gw.by_count]))
    key_named_exprs = [NamedExpr(e.name, e) for e in by_exprs]
    key_schema = {e.name: child_ir.schema[e.name] for e in by_exprs}

    all_scalar_named: list[NamedExpr] = []
    seen: set[str] = set()
    for gw_node in gw_nodes:
        reductions, unary_ops = gw_node._split_named_expr()
        assert not any(unary_ops.values()), "unary window ops not allowed here"
        for ne in reductions:
            if ne.name in seen:
                continue
            all_scalar_named.append(ne)
            seen.add(ne.name)

    name_gen = unique_names(child_ir.schema.keys())
    decompositions = [
        decompose(ne.name, ne.value, names=name_gen) for ne in all_scalar_named
    ]
    selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = combine(
        *decompositions
    )
    assert not need_preshuffle, (
        "Scalar AllGather path does not support aggregations requiring pre-shuffle"
    )

    pwise_schema = dict(key_schema) | {
        ne.name: ne.value.dtype for ne in piecewise_exprs
    }
    piecewise_ir = GroupBy(
        pwise_schema,
        key_named_exprs,
        piecewise_exprs,
        False,  # noqa: FBT003
        None,
        child_ir,
    )

    reduction_key_exprs = [
        NamedExpr(ne.name, Col(pwise_schema[ne.name], ne.name))
        for ne in key_named_exprs
    ]
    reduction_schema = {
        ne.name: ne.value.dtype
        for ne in itertools.chain(reduction_key_exprs, reduction_exprs)
    }
    reduction_ir = GroupBy(
        reduction_schema,
        reduction_key_exprs,
        reduction_exprs,
        False,  # noqa: FBT003
        None,
        piecewise_ir,
    )

    select_key_exprs = [
        NamedExpr(ne.name, Col(reduction_schema[ne.name], ne.name))
        for ne in key_named_exprs
    ]
    select_schema = {
        ne.name: ne.value.dtype
        for ne in itertools.chain(select_key_exprs, selection_exprs)
    }
    agg_select_ir = Select(
        select_schema,
        [*select_key_exprs, *selection_exprs],
        False,  # noqa: FBT003
        reduction_ir,
    )

    return piecewise_ir, reduction_ir, agg_select_ir


class Over(IR):
    """Window over() IR node for the streaming runtime."""

    __slots__ = ("exprs", "is_scalar", "key_indices")
    _preserves_output_order: ClassVar[bool] = True
    _non_child: ClassVar[tuple[str, ...]] = (
        "schema",
        "key_indices",
        "is_scalar",
        "exprs",
    )
    _n_non_child_args: ClassVar[int] = 1
    key_indices: tuple[int, ...]
    is_scalar: bool
    exprs: tuple[NamedExpr, ...]

    def __init__(
        self,
        schema: Schema,
        key_indices: tuple[int, ...],
        is_scalar: bool,  # noqa: FBT001
        exprs: tuple[NamedExpr, ...],
        input_ir: IR,
    ):
        assert len(key_indices) > 0, "Over node requires at least one partition-by key"
        self.schema = schema
        self.key_indices = key_indices
        self.is_scalar = is_scalar
        self.exprs = exprs
        self._non_child_args = (exprs,)
        self.children = (input_ir,)

    @classmethod
    def do_evaluate(
        cls,
        exprs: tuple[NamedExpr, ...],
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate window expressions against df."""
        # At evaluation time Over is just a Select with should_broadcast=True;
        # the window-specific work lives in the GroupedWindow expressions.
        return Select.do_evaluate(exprs, True, df, context=context)  # noqa: FBT003


def _is_scalar_grouped_window(expr: GroupedWindow) -> bool:
    """Return True if this GroupedWindow can use the scalar broadcast path."""
    reductions, unary_ops = expr._split_named_expr()
    if any(unary_ops.values()):
        return False
    if not all(isinstance(c, Col) for c in expr.children[: expr.by_count]):
        return False
    return all(
        isinstance(ne.value, Len)
        or (isinstance(ne.value, Agg) and ne.value.name in _DECOMPOSABLE_AGG_NAMES)
        for ne in reductions
    )


def _extract_over_shuffle_indices(
    expr: GroupedWindow, child_schema: Schema
) -> tuple[int, ...] | None:
    """
    Return partition-by column indices in ``child_schema``, or None.

    Returns None when any partition-by expression is not a plain column
    reference (the multi-partition path only supports Col keys today).
    """
    by_children = expr.children[: expr.by_count]
    if not all(isinstance(c, Col) for c in by_children):
        return None
    return names_to_indices(
        tuple(cast("Col", c).name for c in by_children), child_schema
    )


def _decompose_grouped_window_node(
    expr: GroupedWindow,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Build an Over IR node wrapping a single GroupedWindow expression.

    Every GroupedWindow becomes its own Over here; co-keyed Overs are
    fused together later by select fusion so the actor evaluates all
    window expressions in one pass.

    Returns
    -------
    Expr
        A ``Col`` referencing the Over node's output column, suitable
        for substitution into the enclosing expression.
    IR
        The new ``Over`` IR node.
    MutableMapping[IR, PartitionInfo]
        ``partition_info`` augmented with an entry for the new node.
    """
    indices = _extract_over_shuffle_indices(expr, input_ir.schema)
    if indices is None:
        # TODO: support non-Col partition-by keys on the multi-partition
        # paths. Today the hash shuffle layer rejects expression keys, and
        # the scalar-aggregation broadcast path builds its piecewise
        # groupby from Col by-children directly. Supporting expression
        # keys would require lowering them to columns in the input first.
        raise NotImplementedError(
            "GroupedWindow with non-Col partition-by keys "
            "is not supported for multiple partitions."
        )
    is_scalar = _is_scalar_grouped_window(expr)
    col_name = next(names)
    over_node = Over(
        {col_name: expr.dtype},
        indices,
        is_scalar,
        (NamedExpr(col_name, expr),),
        input_ir,
    )
    partition_info[over_node] = partition_info[input_ir]
    return Col(expr.dtype, col_name), over_node, partition_info


def _fuse_over_nodes(
    selections: list[Select],
    partition_info: MutableMapping[IR, PartitionInfo],
) -> tuple[list[Select], MutableMapping[IR, PartitionInfo]]:
    """
    Fuse per-expression Over nodes that share the same grouping key.

    Selects sharing the Over's input IR are absorbed into the merged Over
    so the actor produces the full output schema in one shuffle pass. The
    grouping key is ``(key_indices, is_scalar, input_ir)``.

    Returns
    -------
    list[Select]
        The rewritten selections: one merged ``Select`` per Over group,
        followed by any selections that were neither part of an Over
        group nor absorbed into one.
    MutableMapping[IR, PartitionInfo]
        ``partition_info`` augmented with entries for the merged Over
        nodes and merged Select nodes introduced by the rewrite.
    """
    over_groups: defaultdict[
        tuple[tuple[int, ...], bool, IR], list[tuple[Select, Over]]
    ] = defaultdict(list)
    passthrough: list[Select] = []

    for sel in selections:
        child = sel.children[0]
        if isinstance(child, Over):
            input_ir = child.children[0]
            over_groups[(child.key_indices, child.is_scalar, input_ir)].append(
                (sel, child)
            )
        else:
            passthrough.append(sel)

    if not over_groups:
        return selections, partition_info

    result: list[Select] = []
    for (key_indices, is_scalar, input_ir), group in over_groups.items():
        pi = partition_info[group[0][1]]

        absorbed: list[Select] = []
        remaining: list[Select] = []
        for s in passthrough:
            (absorbed if s.children[0] == input_ir else remaining).append(s)
        passthrough = remaining

        over_exprs = tuple(
            itertools.chain(
                *(s.exprs for s in absorbed),
                *(over.exprs for _, over in group),
            )
        )
        merged_over = Over(
            {ne.name: ne.value.dtype for ne in over_exprs},
            key_indices,
            is_scalar,
            over_exprs,
            input_ir,
        )
        partition_info[merged_over] = pi
        this_group = {*absorbed, *(sel for sel, _ in group)}
        outer_exprs = tuple(
            itertools.chain.from_iterable(
                s.exprs for s in selections if s in this_group
            )
        )
        outer_schema = {ne.name: ne.value.dtype for ne in outer_exprs}

        merged_sel = Select(outer_schema, outer_exprs, True, merged_over)  # noqa: FBT003
        partition_info[merged_sel] = pi
        result.append(merged_sel)

    result.extend(passthrough)
    return result, partition_info
