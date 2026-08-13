# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.dsl import expr
from cudf_polars.dsl.expr import Col, Len
from cudf_polars.dsl.ir import Empty, HConcat, HStack, Projection, Scan, Select, Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.expressions import (
    decompose_expr_graph,
    make_expr_decomposer,
)
from cudf_polars.streaming.io import StreamingScan
from cudf_polars.streaming.over import _fuse_over_nodes
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.utils import (
    _contains_unsupported_fill_strategy,
    _dynamic_planning_on,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.parallel import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions


def _hstack_chain_to_select(ir: Select) -> Select | None:
    """
    Inline a CSE HStack chain into a flat Select, or return None.

    Only fires when the chain has HStack(should_broadcast=False). Pure HStack(True)
    chains are left for rec(child) — their col refs may live in non-traversable expr
    internals (e.g. GroupedWindow.named_aggs) that _sub_expr cannot reach.
    """
    hstack_chain: list[HStack] = []
    current: IR = ir.children[0]
    all_broadcasted = True
    while isinstance(current, HStack):
        if not current.should_broadcast:
            all_broadcasted = False
        hstack_chain.append(current)
        current = current.children[0]
    base_input = current

    if all_broadcasted:
        return None

    col_defs: dict[str, expr.Expr] = {}
    for hstack in reversed(hstack_chain):
        # with_columns semantics: snapshot so expressions see the input frame,
        # not each other's outputs. CSE placeholders use col_defs directly since
        # later placeholders may reference earlier ones.
        defs = dict(col_defs) if hstack.should_broadcast else col_defs
        for ne in hstack.columns:
            col_defs[ne.name] = _sub_expr(ne.value, defs)

    new_exprs = tuple(
        expr.NamedExpr(ne.name, _sub_expr(ne.value, col_defs)) for ne in ir.exprs
    )
    return Select(ir.schema, new_exprs, ir.should_broadcast, base_input)


@lower_ir_node.register(Projection)
def _(
    ir: Projection, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Always convert Projection(HStack) to Select(HStack)
    # so we can deal with HStack decomposition in one place.
    if isinstance(ir.children[0], HStack):
        expr_from_schema = tuple(
            expr.NamedExpr(name, Col(dtype, name)) for name, dtype in ir.schema.items()
        )
        return lower_ir_node(
            Select(ir.schema, expr_from_schema, True, ir.children[0]),  # noqa: FBT003
            rec,
        )

    # Partition-wise default - Import here to avoid circular import
    from cudf_polars.streaming.parallel import _lower_ir_pwise

    return _lower_ir_pwise(ir, rec, preserve_partitioning=True)


def decompose_select(
    select_ir: Select,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose a multi-partition Select operation.

    Parameters
    ----------
    select_ir
        The original Select operation to decompose.
        This object has not been reconstructed with
        ``input_ir`` as its child yet.
    input_ir
        The lowered child of ``select_ir``. This object
        will be decomposed into a "partial" selection
        for each element of  ``select_ir.exprs``.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    new_ir, partition_info
        The rewritten Select node, and a mapping from
        unique nodes in the new graph to associated
        partitioning information.

    Notes
    -----
    This function uses ``decompose_expr_graph`` to further
    decompose each element of  ``select_ir.exprs``.

    See Also
    --------
    decompose_expr_graph
    """
    # Collect partial selections
    selections: list[Select] = []
    name_generator = unique_names(
        (*(ne.name for ne in select_ir.exprs), *input_ir.schema.keys())
    )
    # Share one decomposer across all NamedExprs so structurally identical
    # sub-expressions (e.g. CSE'd aggregations) are computed only once.
    shared_mapper = make_expr_decomposer(
        input_ir,
        partition_info,
        config_options,
        name_generator,
    )
    for ne in select_ir.exprs:
        # Decompose this partial expression
        new_ne, partial_input_ir, _partition_info = decompose_expr_graph(
            ne,
            mapper=shared_mapper,
        )
        pi = _partition_info[partial_input_ir]
        partial_input_ir = Select(
            {ne.name: ne.value.dtype},
            [new_ne],
            True,  # noqa: FBT003
            partial_input_ir,
        )
        _partition_info[partial_input_ir] = pi
        partition_info.update(_partition_info)
        selections.append(partial_input_ir)

    # Concatenate partial selections
    new_ir: Select | HConcat
    selections, partition_info = _fuse_over_nodes(selections, partition_info)
    selections, partition_info = _fuse_simple_reductions(
        selections,
        partition_info,
    )
    if len(selections) > 1:
        new_ir = HConcat(
            select_ir.schema,
            True,  # noqa: FBT003
            False,  # noqa: FBT003
            *selections,
        )
        partition_info[new_ir] = PartitionInfo(
            count=max(partition_info[c].count for c in selections)
        )
    else:
        new_ir = selections[0]

    return new_ir, partition_info


def _fuse_simple_reductions(
    decomposed_select_irs: Sequence[Select],
    pi: MutableMapping[IR, PartitionInfo],
) -> tuple[list[Select], MutableMapping[IR, PartitionInfo]]:
    """
    Fuse simple reductions that are part of the same Select node.

    Parameters
    ----------
    decomposed_select_irs
        The decomposed Select nodes.
    pi
        Partition information.

    Returns
    -------
    fused_select_irs, pi
        The new Select nodes, and the updated partition information.
    """
    # After a Select node is decomposed, it will be broken into
    # one or more Select nodes that each target a different
    # named expression. In some cases, one or more of these
    # decomposed select nodes will be simple reductions that
    # *should* be performed at the same time. Each "simple"
    # reduction will have the following pattern:
    #
    #   # Partition-wise column selection (select_c)
    #   Select(
    #     # Outer Agg selection (select_b)
    #     Select(
    #       # Repartition to 1 (repartition)
    #       Repartition(
    #         # Inner Agg selection (select_a)
    #         Select(
    #           ...
    #         )
    #       )
    #     )
    #   )
    #
    # We need to fuse these simple reductions together to
    # avoid unnecessary memory pressure.

    # If there is only one decomposed_select_ir, return it
    if len(decomposed_select_irs) == 1:
        return list(decomposed_select_irs), pi

    fused_select_c_exprs = []
    fused_select_c_schema: Schema = {}

    # Find reduction groups
    reduction_groups: defaultdict[IR, list[Select]] = defaultdict(list)
    for select_c in decomposed_select_irs:
        # Final expressions and schema must be included in
        # the fused select_c node even if this specific
        # selection is not a simple reduction.
        fused_select_c_exprs.extend(list(select_c.exprs))
        fused_select_c_schema |= select_c.schema

        if (
            isinstance((select_b := select_c.children[0]), Select)
            and pi[select_b].count == 1
            and isinstance(repartition := select_b.children[0], Repartition)
            and pi[repartition].count == 1
            and isinstance(select_a := repartition.children[0], Select)
        ):
            # We have a simple reduction that may be
            # fused with other simple reductions
            # sharing the same root.
            reduction_root = select_a.children[0]
            reduction_groups[reduction_root].append(select_c)
        else:
            # Not a simple reduction.
            # This selection becomes it own "group".
            reduction_groups[select_c].append(select_c)

    new_decomposed_select_irs: list[IR] = []
    # Schemas already projected to final output names.
    already_projected: Schema = {}
    for root_ir, group in reduction_groups.items():
        if len(group) > 1:
            # When all members share the same select_b (produced by a shared
            # CachingVisitor), the reduction is already computed once. Just
            # combine their select_c exprs into a single node referencing the
            # shared select_b rather than building a new fused chain.
            first_select_b = group[0].children[0]
            if all(s.children[0] is first_select_b for s in group[1:]):
                combined_schema: Schema = {}
                combined_exprs: list[expr.NamedExpr] = []
                for s in group:
                    combined_schema |= s.schema
                    combined_exprs.extend(s.exprs)
                shared_select_c = Select(
                    combined_schema,
                    combined_exprs,
                    True,  # noqa: FBT003
                    first_select_b,
                )
                pi[shared_select_c] = pi[first_select_b]
                new_decomposed_select_irs.append(shared_select_c)
                already_projected |= combined_schema
                continue

            # Fuse simple-aggregation group (distinct select_b nodes)
            fused_select_b_exprs = []
            fused_select_a_exprs = []
            fused_select_b_schema: Schema = {}
            fused_select_a_schema: Schema = {}
            seen_select_a_ids: set[int] = set()
            for select_c in group:
                select_b = select_c.children[0]
                assert isinstance(select_b, Select), (
                    f"Expected Select, got {type(select_b)}"
                )
                select_a = select_b.children[0].children[0]
                assert isinstance(select_a, Select), (
                    f"Expected Select, got {type(select_a)}"
                )
                if (
                    id(select_a) in seen_select_a_ids
                ):  # pragma: no cover; only rapidsmpf hits this
                    continue  # shared by CachingVisitor — already contributed
                seen_select_a_ids.add(id(select_a))
                fused_select_b_exprs.extend(list(select_b.exprs))
                fused_select_b_schema |= select_b.schema
                fused_select_a_exprs.extend(list(select_a.exprs))
                fused_select_a_schema |= select_a.schema
            fused_select_a = Select(
                fused_select_a_schema,
                fused_select_a_exprs,
                True,  # noqa: FBT003
                root_ir,
            )
            pi[fused_select_a] = PartitionInfo(count=pi[root_ir].count)
            fused_repartition = Repartition(fused_select_a_schema, fused_select_a)
            pi[fused_repartition] = PartitionInfo(count=1)
            fused_select_b = Select(
                fused_select_b_schema,
                fused_select_b_exprs,
                True,  # noqa: FBT003
                fused_repartition,
            )
            pi[fused_select_b] = PartitionInfo(count=1)
            new_decomposed_select_irs.append(fused_select_b)
        else:
            # Nothing to fuse for this group, but we still need to add the
            # inner IR (children[0], ie. select_b) rather than the outer select_c.
            # fused_select_c_exprs is built from select_c.exprs, which reference
            # intermediate column names in select_c.children[0].schema, not the
            # final output names in select_c.schema.
            new_decomposed_select_irs.append(group[0].children[0])

    # If any aggregations were fused, we must concatenate
    # the results and apply the final (fused) "c" selection,
    # otherwise we may mess up the ordering of the columns.
    if len(new_decomposed_select_irs) < len(decomposed_select_irs):
        # Compute schema from actual children (intermediate columns)
        hconcat_schema: Schema = {}
        for ir in new_decomposed_select_irs:
            hconcat_schema |= ir.schema
        new_hconcat = HConcat(
            hconcat_schema,
            True,  # noqa: FBT003
            False,  # noqa: FBT003
            *new_decomposed_select_irs,
        )
        count = max(pi[c].count for c in new_decomposed_select_irs)
        pi[new_hconcat] = PartitionInfo(count=count)

        final_exprs = [
            expr.NamedExpr(ne.name, Col(already_projected[ne.name], ne.name))
            if ne.name in already_projected
            else ne
            for ne in fused_select_c_exprs
        ]
        fused_select_c = Select(
            fused_select_c_schema,
            final_exprs,
            True,  # noqa: FBT003
            new_hconcat,
        )
        pi[fused_select_c] = PartitionInfo(count=count)
        return [fused_select_c], pi

    return list(decomposed_select_irs), pi


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if isinstance(ir.children[0], HStack):
        # When the child is a CSE placeholder chain (has HStack(False)), normalize to a
        # flat Select over base_input so decompose_select always gets a chance to run.
        # Pure HStack(True) chains fall through to rec(child) below.
        if (normalized := _hstack_chain_to_select(ir)) is not None:
            return lower_ir_node(normalized, rec)
    elif all(isinstance(e.value, Col) and e.name == e.value.name for e in ir.exprs):
        # Fast path: Equivalent to Projection.
        # We can only do this if the child is NOT HStack.
        return lower_ir_node(
            Projection(ir.schema, ir.children[0]),
            rec,
        )

    child, partition_info = rec(ir.children[0])
    pi = partition_info[child]

    config_options = rec.state["config_options"]
    single_partition = pi.count == 1 and not _dynamic_planning_on(config_options)

    if not single_partition and _contains_unsupported_fill_strategy(
        [e.value for e in ir.exprs]
    ):
        return _lower_ir_fallback(
            ir,
            rec,
            msg=(
                "fill_null with strategy other than 'zero' or 'one' is not supported "
                "for multiple partitions; falling back to in-memory evaluation."
            ),
        )

    # Fast count optimization - reads parquet metadata only, works regardless of partitioning
    scan_child: Scan | None = None
    if Select._is_len_expr(ir.exprs):
        if (
            isinstance(child, Union)
            and len(child.children) == 1
            and isinstance(child.children[0], Scan)
        ):
            # Task engine case
            scan_child = child.children[0]
        elif isinstance(child, StreamingScan):  # pragma: no cover
            # RapidsMPF case
            scan_child = child.base_scan
        elif isinstance(child, Scan):  # pragma: no cover; Requires rapidsmpf runtime
            # Legacy task-engine case
            scan_child = child

    if scan_child and scan_child.predicate is None and scan_child.typ == "parquet":
        # Special Case: Fast count.
        # We can't use prefetched file metadata here, because we're in lowering,
        # not execution, so we don't have an IRExecutionContext with the prefetched
        # file metadata yet.
        count = Scan._get_parquet_row_count_from_metadata(
            scan_child.paths,
            scan_child.skip_rows,
            scan_child.n_rows,
            dataclasses.replace(
                scan_child.parquet_options, prefetch_file_metadata=False
            ),
            None,
        )
        dtype = ir.exprs[0].value.dtype

        lit_expr = expr.LiteralColumn(
            dtype, pl.Series(values=[count], dtype=dtype.polars_type)
        )
        named_expr = expr.NamedExpr(ir.exprs[0].name or "len", lit_expr)

        # Use Empty as the input so the streaming network's metadata flows
        # duplicated=True end to end. Without that, the literal expression
        # would be evaluated chunkwise over child and emit one row per input
        # chunk instead of a single row globally.
        input_ir: IR = Empty({})
        new_node = Select(
            {named_expr.name: named_expr.value.dtype},
            [named_expr],
            should_broadcast=True,
            df=input_ir,
        )
        partition_info[input_ir] = partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    if not any(
        isinstance(expr, (Col, Len)) for expr in traversal([e.value for e in ir.exprs])
    ):
        # Special Case: Selection does not depend on any columns.
        new_node = ir.reconstruct([input_ir := Empty({})])
        partition_info[input_ir] = partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    # Check for non-pointwise expressions (e.g., aggregations)
    has_non_pointwise = not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.exprs])
    )

    # Decompose non-pointwise expressions
    if has_non_pointwise and not single_partition:
        # Special Case: Non-pointwise expressions requiring global aggregation.
        try:
            # Try decomposing the underlying expressions.
            # This inserts Repartition nodes to ensure global aggregation.
            return decompose_select(
                ir,
                child,
                partition_info,
                config_options,
            )
        except NotImplementedError:
            return _lower_ir_fallback(
                ir, rec, msg="This selection is not supported for multiple partitions."
            )

    new_node = ir.reconstruct([child])
    partition_info[new_node] = pi
    return new_node, partition_info


def _sub_expr(e: expr.Expr, subs: dict[str, expr.Expr]) -> expr.Expr:
    if isinstance(e, Col) and e.name in subs:
        return subs[e.name]
    new_children = [_sub_expr(c, subs) for c in e.children]
    if all(new is old for new, old in zip(new_children, e.children, strict=True)):
        return e
    return e.reconstruct(new_children)


def _inline_hstack_false(ir: IR) -> IR:
    """Inline any HStack(should_broadcast=False) CSE chain in ir's first child."""
    if not ir.children:
        return ir
    child = ir.children[0]
    cse_map: dict[str, expr.Expr] = {}
    current = child
    while isinstance(current, HStack) and not current.should_broadcast:
        for ne in current.columns:
            cse_map.setdefault(ne.name, ne.value)
        current = current.children[0]
    base_input = current
    if not cse_map or child is base_input:
        return ir
    if isinstance(ir, HStack):  # pragma: no cover
        new_cols = tuple(
            expr.NamedExpr(ne.name, _sub_expr(ne.value, cse_map)) for ne in ir.columns
        )
        new_schema = {
            **base_input.schema,
            **{ne.name: ne.value.dtype for ne in new_cols},
        }
        return HStack(new_schema, new_cols, ir.should_broadcast, base_input)
    if isinstance(ir, Select):
        new_exprs = tuple(
            expr.NamedExpr(ne.name, _sub_expr(ne.value, cse_map)) for ne in ir.exprs
        )
        return Select(ir.schema, new_exprs, ir.should_broadcast, base_input)
    return ir
