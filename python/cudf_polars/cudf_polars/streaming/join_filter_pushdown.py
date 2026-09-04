# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Rewrite a plan, inserting prefilters in join DAGs.

For a supported inner equijoin, this optimization tries to use the join-key
values produced by one input to reduce the size of the other input before
the original join. In relational notation, a simple rewrite is::

    left join[left.key = right.key] right

        ->

    (left semijoin[left.key = right.key] project(right.key))
       join[left.key = right.key] right

In this example, the right hand table is selected to pre-filter the left
table before performing the inner join.

The implementation uses the following terms:

``column lineage``
    A chain from a named output column towards columns in its input subplan.
    Each step guarantees that every value in the output column also appears in
    the referenced child column, although row order and multiplicity are not
    preserved and the child may contain additional values.
``child edge``
    One particular parent-to-child position in the IR DAG. The same child node
    may occur on more than one edge, so a lineage records child indices and a
    rewrite follows the resulting edge path to change only the chosen
    occurrence.
``target``
    The side of the join to filter.
``domain``
    The side of the join used to provide key values for the filtering of
    ``target``.
``producer``
    A node on a column lineage, together with the column name at that node and
    its edge path from the join input. So termed because it "produces" the
    key values participating in the join.
``source cost``
    An estimate of the cost required to materialize a producer. This guards
    against treating a small intermediate result as a cheap domain when
    producing it requires scanning large inputs.
``constraint domain``
    Selective values of another join key from the target input, used to reduce
    the domain before deriving the values that will filter the target.
``simple candidate``
    A rewrite that projects one domain join key and uses it to filter the
    corresponding target key directly.
``composite candidate``
    For a multi-key join, a rewrite that first semi-joins the domain using the
    constraint domain, then projects the reduced domain's key used to filter
    the target.

Plan rewrite has three stages. ``analyze_plan`` gathers row estimates, source
scan facts, selective nodes, and column value-domain lineages.
Candidate selection consumes those facts and returns a decision.
``apply_candidate`` then constructs the selected semi-join rewrite.

Row estimates, selectivity propagation, thresholds, and candidate scores are
only heuristics for deciding whether a safe rewrite is likely to improve
execution. Poor estimates can choose an unprofitable rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    ConditionalJoin,
    DataFrameScan,
    Distinct,
    Filter,
    GroupBy,
    HStack,
    Join,
    Projection,
    Rolling,
    Scan,
    Select,
    Slice,
    Sort,
    Union,
)
from cudf_polars.dsl.tracing import Scope, log
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    collect_refcount,
    post_traversal,
    reuse_if_unchanged,
    traversal,
)
from cudf_polars.dsl.utils.column_domain import (
    ColumnLineage,
    ColumnRef,
    column_domain_bindings,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.typing import GenericTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


DomainScore: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class SourceFacts:
    """Source-derived facts for an IR node."""

    cost: int | None
    is_single_source: bool


@dataclass(frozen=True)
class _Producer:
    """A subtree and its bound column names at an insertion point."""

    node: IR
    columns: tuple[str, ...]
    rows: int
    cost: int
    is_single_source: bool
    path: tuple[int, ...] = ()
    """Child-edge path from the candidate root to ``node``."""

    @property
    def column(self) -> str:
        """First bound column in the producer."""
        return self.columns[0]

    @property
    def domain_score(self) -> DomainScore:
        """Scoring function for a domain."""
        return (self.cost, self.rows, len(self.node.schema))


@dataclass(frozen=True)
class SimpleCandidate:
    """A direct key-domain prefilter candidate."""

    mode = "simple"
    target_side: Literal["left", "right"]
    target: _Producer
    target_key: expr.Col
    domain: _Producer
    domain_key: expr.Col

    @property
    def score(self) -> tuple[int, DomainScore]:
        """Rank after composite candidates, then by domain cost."""
        return (1, self.domain.domain_score)


@dataclass(frozen=True)
class CompositeCandidate:
    """A key-domain prefilter constrained by another join key."""

    mode = "composite"
    target_side: Literal["left", "right"]
    target: _Producer
    target_key: expr.Col
    domain: _Producer
    domain_key: expr.Col
    constraint_domain: _Producer
    domain_constraint_key: expr.Col
    target_constraint_key: expr.Col

    @property
    def score(self) -> tuple[int, DomainScore, DomainScore]:
        """Prefer cheaper constraint and domain inputs."""
        return (0, self.constraint_domain.domain_score, self.domain.domain_score)


@dataclass(frozen=True)
class _AggregateReuseCandidate:
    """A join detail side that can be replaced by an existing aggregate domain."""

    join: Join
    detail_side: Literal["left", "right"]
    value_column: expr.Col
    replacement: IR
    domain_node_type: str
    replacement_rows: int
    replacement_cost: int


Candidate: TypeAlias = SimpleCandidate | CompositeCandidate
DecisionReason: TypeAlias = Literal[
    "applied",
    "maintain_order",
    "no_profitable_domain",
    "non_column_join_key",
    "not_inner_join",
    "sliced_join",
]


@dataclass(frozen=True)
class Decision:
    """Result of considering a join for a domain prefilter."""

    reason: DecisionReason
    candidate: Candidate | None = None


@dataclass(frozen=True)
class PlanFacts:
    """Facts derived in one bottom-up traversal of an IR DAG."""

    row_estimates: Mapping[IR, int | None]
    source_facts: Mapping[IR, SourceFacts]
    selective_nodes: frozenset[IR]
    column_lineages: Mapping[ColumnRef, ColumnLineage]
    refcounts: Mapping[IR, int]


class _RewriteState(TypedDict):
    """State shared by the join-domain prefilter DAG rewrite."""

    threshold: float
    trace: bool
    stats: StatsCollector
    facts: PlanFacts


def analyze_plan(ir: IR, stats: StatsCollector) -> PlanFacts:
    """
    Derive row, selectivity, and column-domain facts for an IR DAG.

    Parameters
    ----------
    ir
        Root node to gather facts for.
    stats
        Pre-populated statistics

    Returns
    -------
    Gather facts about the plan.
    """
    row_estimates: dict[IR, int | None] = {}
    source_facts: dict[IR, SourceFacts] = {}
    source_nodes: dict[IR, frozenset[IR]] = {}
    selective_nodes: set[IR] = set()
    column_lineages: dict[ColumnRef, ColumnLineage] = {}
    refcounts = collect_refcount([ir])

    for node in post_traversal([ir]):
        if isinstance(node, (Scan, DataFrameScan)):
            source_info = stats.scan_stats.get(node)
            rows = None if source_info is None else source_info.row_count
            if rows is None and isinstance(node, DataFrameScan):
                rows = node.df.shape()[0]
        elif isinstance(node, (Select, Projection, HStack, Filter, Distinct, GroupBy)):
            rows = row_estimates[node.children[0]]
        elif isinstance(node, Join):
            rows = _estimate_join_rows(
                node.options[0],
                row_estimates[node.children[0]],
                row_estimates[node.children[1]],
            )
        else:
            child_estimates = [
                estimate
                for child in node.children
                if (estimate := row_estimates[child]) is not None
            ]
            rows = max(child_estimates, default=None)
        row_estimates[node] = rows

        if isinstance(node, (Scan, DataFrameScan)):
            sources: frozenset[IR] = frozenset((node,))
        else:
            sources = frozenset(
                source for child in node.children for source in source_nodes[child]
            )
        source_nodes[node] = sources
        source_rows = [
            source_rows
            for source in sources
            if (source_rows := row_estimates[source]) is not None and source_rows > 0
        ]
        source_facts[node] = SourceFacts(
            cost=sum(source_rows) if source_rows else rows,
            is_single_source=len(sources) == 1,
        )

        if (
            (isinstance(node, Scan) and node.predicate is not None)
            or isinstance(node, Filter)
            or any(child in selective_nodes for child in node.children)
        ):
            selective_nodes.add(node)

        bindings = column_domain_bindings(node)
        for name in node.schema:
            column = ColumnRef(node, name)
            binding = bindings.get(name)
            if binding is None:
                source_lineage = None
                source_child_index = None
            else:
                source_child_index = binding.child_index
                source = ColumnRef(
                    node.children[source_child_index],
                    binding.name,
                )
                source_lineage = column_lineages[source]
            column_lineages[column] = ColumnLineage(
                column, source_lineage, source_child_index
            )

    return PlanFacts(
        row_estimates=row_estimates,
        source_facts=source_facts,
        selective_nodes=frozenset(selective_nodes),
        column_lineages=column_lineages,
        refcounts=refcounts,
    )


def blocks_pushdown(node: IR, facts: PlanFacts) -> bool:
    """
    Return whether a node blocks filter pushdown.

    Parameters
    ----------
    node
        Node to check.
    facts
        Facts about the plan.

    Returns
    -------
    bool
        True if a semijoin cannot be pushed past this node, otherwise False.
    """
    # TODO: Need better cost model to handle nodes that are shared. Pushing
    # a filter into a shared node will typically mean that it is no longer
    # shared, since the same filter will not come from every consumer.
    return facts.refcounts[node] > 1 or (
        # TODO: Distinct and Rolling only block pushdown in some
        # circumstances, but we'd need to make the logic more complicated:
        # - We can push through distinct if the filter applies to the columns
        #   that are being used to determine distinct rows
        # - We can push through rolling if the filter applies to the
        #   groupby keys.
        # TODO: We can push through an unsliced Union, but need to
        # distribute the filter onto every child.
        isinstance(node, (Distinct, Rolling, Slice, Union))
        # Can't push through anything that is sliced.
        or (isinstance(node, (GroupBy, Sort)) and node.zlice is not None)
        or (isinstance(node, (ConditionalJoin, Join)) and node.options[2] is not None)
    )


def semijoin_pushdown_candidates(
    facts: PlanFacts, root: IR, column: str
) -> Iterator[tuple[ColumnRef, tuple[int, ...]]]:
    """
    Yield column domain lineage providing valid locations for semijoin pushdown.

    Parameters
    ----------
    facts
        Gathered facts about the plan
    root
        Root node to search from
    column
        Name of column we're finding the lineage of.

    Returns
    -------
    Iterator
        Of valid insertion points and their child-edge paths from ``root``.
    """
    try:
        lineage = facts.column_lineages[ColumnRef(root, column)]
    except KeyError:
        return
    path: tuple[int, ...] = ()
    while True:
        yield lineage.column, path
        source = lineage.source
        source_child_index = lineage.source_child_index
        if blocks_pushdown(lineage.column.node, facts) or source is None:
            return
        assert source_child_index is not None
        path = (*path, source_child_index)
        lineage = source


def optimize_join_filter_pushdown(
    ir: IR,
    stats: StatsCollector,
    config_options: ConfigOptions[StreamingExecutor],
) -> IR:
    """
    Rewrite an IR DAG to apply filter pushdown of keys.

    This optimization pass inspects joins in the DAG and attempts to push a
    prefilter obtained from the keys of one side of the join onto the
    inputs of the other side. This can be highly beneficial at large scale
    since if we have a selective join we can avoid data movement by
    prefiltering before performing the actual join.

    Parameters
    ----------
    ir
        DAG to rewrite.
    stats
        Pre-populated statistics.
    config_options
        Configuration options controlling the rewrite.

    Returns
    -------
    Rewritten DAG.
    """
    options = config_options.executor.join_filter_pushdown
    if options is None:
        return ir
    threshold = options.threshold
    trace = options.trace
    if threshold == 0:
        return ir

    state = _RewriteState(
        threshold=threshold,
        trace=trace,
        stats=stats,
        facts=analyze_plan(ir, stats),
    )
    mapper: GenericTransformer[IR, IR, _RewriteState] = CachingVisitor(
        _rewrite, state=state
    )
    return mapper(ir)


@singledispatch
def _rewrite(node: IR, rec: GenericTransformer[IR, IR, _RewriteState]) -> IR:
    raise AssertionError


@_rewrite.register(IR)
def _(node: IR, rec: GenericTransformer[IR, IR, _RewriteState]) -> IR:
    return reuse_if_unchanged(node, rec)


@_rewrite.register(Join)
def _(node: Join, rec: GenericTransformer[IR, IR, _RewriteState]) -> IR:
    original = node
    rewritten = reuse_if_unchanged(node, rec)
    assert isinstance(rewritten, Join)
    node = rewritten
    if node is original:
        facts = rec.state["facts"]
    else:
        # Child rewrites introduce new semi joins and reconstructed ancestors.
        # Re-analyze that current subtree so parent joins can use the derived
        # selectivity and cardinality when ranking their own candidates.
        facts = analyze_plan(node, rec.state["stats"])
    decision = _select_candidate(
        node,
        rec.state["threshold"],
        facts,
    )
    if rec.state["trace"]:
        _trace_decision(node, rec.state["threshold"], decision)
    if decision.candidate is None:
        return node
    return apply_candidate(node, decision.candidate)


@_rewrite.register(GroupBy)
def _(node: GroupBy, rec: GenericTransformer[IR, IR, _RewriteState]) -> IR:
    original = node
    rewritten = reuse_if_unchanged(node, rec)
    assert isinstance(rewritten, GroupBy)
    node = rewritten
    if node is original:
        facts = rec.state["facts"]
    else:
        facts = analyze_plan(node, rec.state["stats"])
    rewritten, aggregate_candidate = _rewrite_aggregate_domain_reuse(node, facts)
    if rec.state["trace"]:
        _trace_aggregate_reuse(node, aggregate_candidate)
    return rewritten


def apply_candidate(ir: Join, candidate: Candidate) -> IR:
    """Apply a selected join-domain prefilter candidate to a join."""
    left, right = ir.children
    domain = _make_domain(candidate, ir)
    target = candidate.target
    target_filter = _make_semi_join(
        target.node,
        expr.Col(target.node.schema[target.column], target.column),
        domain,
        expr.Col(domain.schema[candidate.domain_key.name], candidate.domain_key.name),
        nulls_equal=ir.options[1],
        suffix=ir.options[3],
    )
    if candidate.target_side == "left":
        left = replace_at_path(left, target.path, target_filter)
    else:
        right = replace_at_path(right, target.path, target_filter)
    return ir.reconstruct((left, right))


def replace_at_path(root: IR, path: Sequence[int], replacement: IR) -> IR:
    """
    Replace a specific child in a DAG starting at root.

    Parameters
    ----------
    root
        Root of DAG to carry out replacement.
    path
        Breadcrumb trail selecting which child at every level to recurse
        into.
    replacement
        Replacement node to return when the path becomes empty.

    Returns
    -------
    IR
        New DAG with the selected child replaced with replacement.

    Notes
    -----
    This specifically does not use replacement by equality so that we can
    disambiguate between shared children in the DAG where we only want to
    replace one.
    """
    if not path:
        return replacement
    index, *path = path
    children = list(root.children)
    children[index] = replace_at_path(children[index], path, replacement)
    return root.reconstruct(children)


def _select_candidate(
    ir: Join,
    threshold: float,
    facts: PlanFacts,
) -> Decision:
    if ir.options[0] != "Inner":
        return Decision(reason="not_inner_join")
    if ir.options[2] is not None:
        return Decision(reason="sliced_join")
    if ir.options[5] != "none":
        return Decision(reason="maintain_order")

    left_keys = _simple_keys(ir.left_on)
    right_keys = _simple_keys(ir.right_on)
    if len(left_keys) != len(ir.left_on) or len(right_keys) != len(ir.right_on):
        return Decision(reason="non_column_join_key")

    candidates: list[Candidate] = []
    left: tuple[Literal["left", "right"], IR, tuple[expr.Col, ...]] = (
        "left",
        ir.children[0],
        left_keys,
    )
    right: tuple[Literal["left", "right"], IR, tuple[expr.Col, ...]] = (
        "right",
        ir.children[1],
        right_keys,
    )
    for (target_side, target_child, target_keys), (
        _,
        domain_child,
        domain_keys,
    ) in ((left, right), (right, left)):
        candidates.extend(
            _composite_candidates(
                target_side,
                target_child,
                domain_child,
                target_keys,
                domain_keys,
                threshold,
                facts,
            )
        )
        candidates.extend(
            _simple_candidates(
                target_side,
                target_child,
                domain_child,
                target_keys,
                domain_keys,
                threshold,
                facts,
            )
        )

    if not candidates:
        return Decision(reason="no_profitable_domain")
    return Decision(reason="applied", candidate=min(candidates, key=lambda c: c.score))


def _rewrite_aggregate_domain_reuse(
    ir: GroupBy, facts: PlanFacts
) -> tuple[GroupBy, _AggregateReuseCandidate | None]:
    """Replace one detail join with a compatible existing aggregate domain."""
    if ir.maintain_order or ir.zlice is not None:
        return ir, None

    aggregate_column = _single_summed_column(ir)
    if aggregate_column is None:
        return ir, None

    candidates = list(
        _aggregate_reuse_candidates(ir.children[0], aggregate_column, facts)
    )
    if not candidates:
        return ir, None

    for candidate in sorted(
        candidates,
        key=lambda item: (item.replacement_rows, item.replacement_cost),
    ):
        child = _replace_on_aggregate_path(
            ir.children[0],
            aggregate_column.name,
            candidate.join,
            candidate.replacement,
        )
        if child is not None:
            rewritten = ir.reconstruct((child,))
            assert isinstance(rewritten, GroupBy)
            return rewritten, candidate
    return ir, None


def _single_summed_column(ir: GroupBy) -> expr.Col | None:
    """Return the sole directly summed, non-key column."""
    if len(ir.agg_requests) != 1:
        return None
    value = ir.agg_requests[0].value
    if not isinstance(value, expr.Agg) or value.name != "sum":
        return None
    if len(value.children) != 1:
        return None
    (child,) = value.children
    if not isinstance(child, expr.Col):
        return None

    if any(
        isinstance(node, expr.Col) and node.name == child.name
        for node in traversal([key.value for key in ir.keys])
    ):
        return None
    return child


def _aggregate_reuse_candidates(
    root: IR,
    summed_column: expr.Col,
    facts: PlanFacts,
) -> Iterable[_AggregateReuseCandidate]:
    """Yield candidates along the exact input lineage of the final sum."""
    bindings = tuple(_exact_column_bindings(root, summed_column.name))
    for index, (node, bound_value_column) in enumerate(bindings):
        if (
            isinstance(node, Join)
            and node.options[0] == "Inner"
            and node.options[2] is None
            and node.options[5] == "none"
        ):
            yield from _aggregate_reuse_candidates_for_join(
                node,
                bound_value_column,
                facts,
            )

        if index + 1 < len(bindings):
            child, input_column = bindings[index + 1]
            if not _aggregate_reuse_edge_is_safe(
                node, bound_value_column, child, input_column
            ):
                return


def _aggregate_reuse_candidates_for_join(
    node: Join,
    summed_column: str,
    facts: PlanFacts,
) -> Iterable[_AggregateReuseCandidate]:
    """Yield aggregate replacements for one join on the sum lineage."""
    left_keys = _simple_keys(node.left_on)
    right_keys = _simple_keys(node.right_on)
    if len(left_keys) != len(node.left_on) or len(right_keys) != len(node.right_on):
        return
    assert len(left_keys) == len(right_keys)

    value_binding = _join_input_binding(node, summed_column)
    if value_binding is None:
        return
    detail_child, detail_value_column = value_binding
    detail_indices = [
        index for index, child in enumerate(node.children) if child is detail_child
    ]
    if len(detail_indices) != 1:
        return
    (detail_index,) = detail_indices
    detail_side: Literal["left", "right"] = "left" if detail_index == 0 else "right"
    detail_keys = left_keys if detail_index == 0 else right_keys
    domain_child = node.children[1 - detail_index]
    domain_keys = right_keys if detail_index == 0 else left_keys
    allowed_columns = {detail_value_column}
    allowed_columns.update(key.name for key in detail_keys)
    bound_detail_columns = set()
    for output_column in node.schema:
        binding = _join_input_binding(node, output_column)
        if binding is not None and binding[0] is detail_child:
            bound_detail_columns.add(binding[1])
    if not bound_detail_columns.issubset(allowed_columns):
        return

    value_column = expr.Col(
        detail_child.schema[detail_value_column], detail_value_column
    )
    replacement_detail = _aggregate_reuse_detail_replacement(
        detail_child,
        domain_child,
        domain_keys,
        detail_keys,
        value_column,
        node.options[1],
        node.options[3],
        facts,
    )
    if replacement_detail is None:
        return
    replacement, domain_node_type, replacement_rows, replacement_cost = (
        replacement_detail
    )
    children = list(node.children)
    children[detail_index] = replacement
    yield _AggregateReuseCandidate(
        join=node,
        detail_side=detail_side,
        value_column=value_column,
        replacement=node.reconstruct(children),
        domain_node_type=domain_node_type,
        replacement_rows=replacement_rows,
        replacement_cost=replacement_cost,
    )


def _aggregate_reuse_detail_replacement(
    detail_child: IR,
    domain_child: IR,
    domain_keys: tuple[expr.Col, ...],
    detail_keys: tuple[expr.Col, ...],
    value_column: expr.Col,
    nulls_equal: bool,  # noqa: FBT001
    suffix: str,
    facts: PlanFacts,
) -> tuple[IR, str, int, int] | None:
    """Build a null-correct aggregate replacement for a semi-filtered detail side."""
    if len(detail_keys) != 1:
        return None
    (detail_key,) = detail_keys
    if isinstance(detail_child, Join) and detail_child.options[0] == "Semi":
        return _aggregate_reuse_detail_replacement_from_semi_detail(
            detail_child,
            detail_key,
            value_column,
            facts,
        )
    if len(domain_keys) != 1:
        return None
    domain = _aggregate_reuse_domain_from_sibling_semi(
        domain_child,
        domain_keys[0],
        detail_child,
        detail_key,
        value_column,
        nulls_equal,
        facts,
    )
    if domain is None:
        return None
    return _aggregate_reuse_detail_replacement_from_domain(
        domain_root=domain.root,
        aggregate_domain=domain.aggregate,
        detail_key=detail_key,
        value_column=value_column,
        nulls_equal=nulls_equal,
        suffix=suffix,
    )


def _aggregate_reuse_detail_replacement_from_semi_detail(
    detail_child: Join,
    detail_key: expr.Col,
    value_column: expr.Col,
    facts: PlanFacts,
) -> tuple[IR, str, int, int] | None:
    """Build an aggregate replacement when the detail side is already semi-filtered."""
    if detail_child.options[2] is not None or detail_child.options[5] != "none":
        return None

    semi_left_keys = _simple_keys(detail_child.left_on)
    semi_right_keys = _simple_keys(detail_child.right_on)
    if len(semi_left_keys) != len(detail_child.left_on) or len(semi_right_keys) != len(
        detail_child.right_on
    ):
        return None
    if semi_left_keys != (detail_key,) or len(semi_right_keys) != 1:
        return None
    (domain_key,) = semi_right_keys

    aggregate_domain = _selective_aggregate_domain(
        detail_child.children[1],
        domain_key,
        detail_child.children[0],
        detail_key,
        value_column,
        facts,
    )
    if aggregate_domain is None:
        return None
    return _aggregate_reuse_detail_replacement_from_domain(
        domain_root=detail_child.children[1],
        aggregate_domain=aggregate_domain,
        detail_key=detail_key,
        value_column=value_column,
        nulls_equal=detail_child.options[1],
        suffix=detail_child.options[3],
    )


@dataclass(frozen=True)
class _AggregateSiblingDomain:
    """An aggregate domain found through a sibling semi join."""

    root: IR
    aggregate: _Producer


def _aggregate_reuse_domain_from_sibling_semi(
    domain_child: IR,
    domain_key: expr.Col,
    detail_child: IR,
    detail_key: expr.Col,
    value_column: expr.Col,
    nulls_equal: bool,  # noqa: FBT001
    facts: PlanFacts,
) -> _AggregateSiblingDomain | None:
    """Find an aggregate domain used to semi-filter the other join side."""
    if not isinstance(domain_child, Join) or domain_child.options[0] != "Semi":
        return None
    if domain_child.options[2] is not None or domain_child.options[5] != "none":
        return None
    if domain_child.options[1] != nulls_equal:
        return None

    semi_left_keys = _simple_keys(domain_child.left_on)
    semi_right_keys = _simple_keys(domain_child.right_on)
    if len(semi_left_keys) != len(domain_child.left_on) or len(semi_right_keys) != len(
        domain_child.right_on
    ):
        return None
    if semi_left_keys != (domain_key,) or len(semi_right_keys) != 1:
        return None
    (aggregate_key,) = semi_right_keys

    aggregate_domain = _selective_aggregate_domain(
        domain_child.children[1],
        aggregate_key,
        detail_child,
        detail_key,
        value_column,
        facts,
    )
    if aggregate_domain is None:
        return None
    return _AggregateSiblingDomain(domain_child.children[1], aggregate_domain)


def _aggregate_reuse_detail_replacement_from_domain(
    *,
    domain_root: IR,
    aggregate_domain: _Producer,
    detail_key: expr.Col,
    value_column: expr.Col,
    nulls_equal: bool,
    suffix: str,
) -> tuple[IR, str, int, int]:
    """Project aggregate key/value columns into a detail-side replacement."""
    domain = aggregate_domain
    aggregate_values = _project_bound_key_and_value(
        domain.node,
        domain.columns[0],
        domain.columns[1],
        detail_key,
        value_column,
    )
    replacement: IR
    if _domain_key_set_is_preserved(domain_root, domain.node, domain.columns[0]):
        replacement = (
            aggregate_values
            if nulls_equal
            else _drop_null_key(aggregate_values, detail_key)
        )
    else:
        replacement = _make_semi_join(
            aggregate_values,
            expr.Col(aggregate_values.schema[detail_key.name], detail_key.name),
            domain_root,
            expr.Col(domain_root.schema[domain.columns[0]], domain.columns[0]),
            nulls_equal=nulls_equal,
            suffix=suffix,
        )
    return replacement, type(domain.node).__name__, domain.rows, domain.cost


def _selective_aggregate_domain(
    root: IR,
    domain_key: expr.Col,
    detail_source: IR,
    detail_key: expr.Col,
    value_column: expr.Col,
    facts: PlanFacts,
) -> _Producer | None:
    """Find a selective aggregate derived from the identical detail source."""
    candidates = []
    for groupby, bound_domain_key in _exact_column_bindings(root, domain_key.name):
        if not isinstance(groupby, GroupBy):
            continue
        if (
            not _same_detail_source(groupby.children[0], detail_source)
            or len(groupby.keys) != 1
        ):
            continue
        (groupby_key,) = groupby.keys
        if (
            groupby_key.name != bound_domain_key
            or not isinstance(groupby_key.value, expr.Col)
            or groupby_key.value != detail_key
        ):
            continue
        aggregate_column = _sum_aggregate_output_column(groupby, value_column)
        if aggregate_column is None:
            continue
        domain = _smallest_selective_node_containing_all(
            root,
            bound_domain_key,
            aggregate_column.name,
            groupby,
            facts,
        )
        if domain is not None:
            candidates.append((domain.rows, domain.cost, domain))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _same_detail_source(left: IR, right: IR) -> bool:
    """Return whether two nodes refer to the same materialized detail source."""
    if left is right:
        return True
    if isinstance(left, Scan) and isinstance(right, Scan):
        return (
            left.typ == right.typ
            and left.reader_options == right.reader_options
            and left.cloud_options == right.cloud_options
            and left.paths == right.paths
            and left.skip_rows == right.skip_rows
            and left.n_rows == right.n_rows
            and left.row_index == right.row_index
            and left.include_file_paths == right.include_file_paths
            and left.predicate == right.predicate
            and left.parquet_options == right.parquet_options
        )
    if isinstance(left, DataFrameScan) and isinstance(right, DataFrameScan):
        return left == right
    return (
        isinstance(left, Cache)
        and isinstance(right, Cache)
        and left.key == right.key
        and left.refcount == right.refcount
        and left.schema == right.schema
    )


def _sum_aggregate_output_column(
    ir: GroupBy, value_column: expr.Col
) -> expr.Col | None:
    """Return the output of a direct sum over the requested value column."""
    for request in ir.agg_requests:
        value = request.value
        if not isinstance(value, expr.Agg) or value.name != "sum":
            continue
        if value.dtype != value_column.dtype or len(value.children) != 1:
            continue
        (child,) = value.children
        if isinstance(child, expr.Col) and child.name == value_column.name:
            if request.name not in ir.schema:
                return None
            return expr.Col(ir.schema[request.name], request.name)
    return None


def _smallest_selective_node_containing_all(
    root: IR,
    key_column: str,
    value_column: str,
    anchor: IR,
    facts: PlanFacts,
) -> _Producer | None:
    """Find the smallest selective node whose columns bind to an anchor."""
    candidates = []
    for node in traversal([root]):
        bound_columns = _bound_aggregate_output_columns(
            node, anchor, key_column, value_column
        )
        if bound_columns is None or node not in facts.selective_nodes:
            continue
        producer = make_producer(node, bound_columns, (), facts)
        if producer is not None:
            candidates.append(
                (producer.rows, producer.cost, len(node.schema), producer)
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _bound_aggregate_output_columns(
    node: IR, anchor: IR, key_column: str, value_column: str
) -> tuple[str, ...] | None:
    """Map an aggregate key and normalized sum output through a domain node."""
    outputs = []
    for anchor_column, binding_fn in (
        (key_column, _exact_column_bindings),
        (value_column, _aggregate_value_bindings),
    ):
        matches = [
            output_column
            for output_column in node.schema
            if any(
                candidate is anchor and bound_column == anchor_column
                for candidate, bound_column in binding_fn(node, output_column)
            )
        ]
        if len(matches) != 1:
            return None
        outputs.append(matches[0])
    return tuple(outputs)


def _aggregate_value_bindings(root: IR, column: str) -> Iterable[tuple[IR, str]]:
    """Yield sum-value lineage through direct bindings and zero-fill normalization."""
    node = root
    while column in node.schema:
        yield node, column
        binding = _input_binding(node, column)
        if binding is None and isinstance(node, Select):
            selected = next((item for item in node.exprs if item.name == column), None)
            binding = _zero_fill_binding(node.children[0], selected)
        if binding is None:
            return
        node, column = binding


def _zero_fill_binding(
    child: IR, expression: expr.NamedExpr | None
) -> tuple[IR, str] | None:
    """Bind Polars' ``sum`` null normalization to its aggregate output."""
    if expression is None or not isinstance(expression.value, expr.UnaryFunction):
        return None
    value = expression.value
    if value.name != "fill_null" or value.options or len(value.children) != 2:
        return None
    source, fill = value.children
    if (
        not isinstance(source, expr.Col)
        or not isinstance(fill, expr.Literal)
        or fill.value != 0
        or source.name not in child.schema
        or source.dtype != child.schema[source.name]
        or fill.dtype != source.dtype
        or value.dtype != source.dtype
    ):
        return None
    return child, source.name


def _project_bound_key_and_value(
    source: IR,
    source_key: str,
    source_value: str,
    output_key: expr.Col,
    output_value: expr.Col,
) -> Select:
    """Project aggregate key and value columns under detail-side names."""
    assert source.schema[source_key] == output_key.dtype
    assert source.schema[source_value] == output_value.dtype
    return Select(
        {output_key.name: output_key.dtype, output_value.name: output_value.dtype},
        (
            expr.NamedExpr(output_key.name, expr.Col(output_key.dtype, source_key)),
            expr.NamedExpr(
                output_value.name, expr.Col(output_value.dtype, source_value)
            ),
        ),
        True,  # noqa: FBT003
        source,
    )


def _domain_key_set_is_preserved(root: IR, domain: IR, column: str) -> bool:
    """Return whether a domain reaches the semi join through row-preserving nodes."""
    for node, _ in _exact_column_bindings(root, column):
        if node is domain:
            return True
        if not isinstance(node, (Cache, HStack, Projection, Select)):
            return False
    return False


def _drop_null_key(source: IR, key: expr.Col) -> Filter:
    """Apply the null-key behavior of a semi join with ``nulls_equal=False``."""
    bool_dtype = DataType(pl.Boolean())
    predicate = expr.NamedExpr(
        "__join_filter_pushdown_key_is_not_null",
        expr.BooleanFunction(
            bool_dtype,
            expr.BooleanFunction.Name.IsNotNull,
            (),
            expr.Col(key.dtype, key.name),
        ),
    )
    return Filter(source.schema, predicate, source)


def _aggregate_reuse_edge_is_safe(
    node: IR,
    output_column: str,
    child: IR,
    input_column: str,
) -> bool:
    """Return whether aggregate values can move across one lineage edge."""
    child_indices = [
        index for index, candidate in enumerate(node.children) if candidate is child
    ]
    if len(child_indices) != 1:
        return False
    (child_index,) = child_indices
    if _input_binding(node, output_column) != (child, input_column):
        return False
    if isinstance(node, (Cache, HStack, Projection, Select)):
        return True
    if not isinstance(node, Join):
        return False
    if (
        node.options[0] != "Inner"
        or node.options[2] is not None
        or node.options[5] != "none"
    ):
        return False
    keys = node.left_on if child_index == 0 else node.right_on
    return not any(
        isinstance(item, expr.Col) and item.name == input_column
        for item in traversal([key.value for key in keys])
    )


def _replace_on_aggregate_path(
    root: IR,
    column: str,
    target: Join,
    replacement: IR,
) -> IR | None:
    """Replace a join only on the direct lineage of an aggregate column."""
    bindings = tuple(_exact_column_bindings(root, column))
    path: list[tuple[IR, int]] = []
    for index, (node, _) in enumerate(bindings):
        if node is target:
            rewritten = replacement
            for parent, child_index in reversed(path):
                children = list(parent.children)
                children[child_index] = rewritten
                rewritten = parent.reconstruct(children)
            return rewritten
        if index + 1 == len(bindings):
            break
        child = bindings[index + 1][0]
        child_indices = [
            child_index
            for child_index, candidate in enumerate(node.children)
            if candidate is child
        ]
        if len(child_indices) != 1:
            return None
        path.append((node, child_indices[0]))
    return None


def _exact_column_bindings(root: IR, column: str) -> Iterable[tuple[IR, str]]:
    """Yield direct output-to-input bindings for a column through a subplan."""
    node = root
    while column in node.schema:
        yield node, column
        binding = _input_binding(node, column)
        if binding is None:
            return
        node, column = binding


def _input_binding(node: IR, column: str) -> tuple[IR, str] | None:
    """Return a proven direct input binding, stopping at ambiguous operations."""
    child = node.children[0] if len(node.children) == 1 else None
    if isinstance(node, Select):
        selected = next((item for item in node.exprs if item.name == column), None)
        return _column_expression_binding(child, selected)
    if isinstance(node, HStack):
        stacked = next((item for item in node.columns if item.name == column), None)
        if stacked is not None:
            return _column_expression_binding(child, stacked)
        return _passthrough_binding(child, column)
    if isinstance(node, GroupBy):
        if node.zlice is not None:
            return None
        key = next((item for item in node.keys if item.name == column), None)
        return _column_expression_binding(child, key)
    if isinstance(node, Join):
        return _join_input_binding(node, column)
    if isinstance(node, Distinct):
        return None if node.zlice is not None else _passthrough_binding(child, column)
    if isinstance(node, Sort):
        return None if node.zlice is not None else _passthrough_binding(child, column)
    if isinstance(node, (Cache, Filter, Projection)):
        return _passthrough_binding(child, column)
    return None


def _column_expression_binding(
    child: IR | None, expression: expr.NamedExpr | None
) -> tuple[IR, str] | None:
    if (
        child is not None
        and expression is not None
        and isinstance(expression.value, expr.Col)
        and expression.value.name in child.schema
    ):
        return child, expression.value.name
    return None


def _passthrough_binding(child: IR | None, column: str) -> tuple[IR, str] | None:
    if child is not None and column in child.schema:
        return child, column
    return None


def _join_input_binding(node: Join, column: str) -> tuple[IR, str] | None:
    if node.options[2] is not None:
        return None
    left, right = node.children
    if node.options[0] in ("Semi", "Anti"):
        return _passthrough_binding(left, column)
    if node.options[0] != "Inner":
        return None
    bindings = []
    if column in left.schema:
        bindings.append((left, column))
    suffix = node.options[3]
    for right_column in right.schema:
        output_column = (
            f"{right_column}{suffix}" if right_column in left.schema else right_column
        )
        if output_column == column and output_column in node.schema:
            bindings.append((right, right_column))
    if len(bindings) == 1:
        return bindings[0]
    return None


def _simple_keys(keys: Sequence[expr.NamedExpr]) -> tuple[expr.Col, ...]:
    return tuple(key.value for key in keys if isinstance(key.value, expr.Col))


def _simple_candidates(
    target_side: Literal["left", "right"],
    target_child: IR,
    domain_child: IR,
    target_keys: tuple[expr.Col, ...],
    domain_keys: tuple[expr.Col, ...],
    threshold: float,
    facts: PlanFacts,
) -> Iterable[SimpleCandidate]:
    for target_key, domain_key in zip(target_keys, domain_keys, strict=True):
        target = _largest_key_source(target_child, target_key.name, facts)
        if target is None:
            continue
        domain = _smallest_key_producer(
            domain_child,
            domain_key.name,
            facts,
            require_selective=True,
        )
        if domain is None:
            continue
        if domain.rows / target.rows > threshold:
            continue
        if contains_node(target.node, domain.node):
            continue
        if domain.is_single_source and has_filtering_semi_ancestor(
            target_child, target.path
        ):
            continue
        if not domain_cost_is_small(domain, target, threshold):
            continue
        yield SimpleCandidate(
            target_side=target_side,
            target=target,
            target_key=target_key,
            domain=domain,
            domain_key=domain_key,
        )


def _composite_candidates(
    target_side: Literal["left", "right"],
    target_child: IR,
    domain_child: IR,
    target_keys: tuple[expr.Col, ...],
    domain_keys: tuple[expr.Col, ...],
    threshold: float,
    facts: PlanFacts,
) -> Iterable[CompositeCandidate]:
    if len(target_keys) < 2:
        return

    for filter_index, (target_key, domain_key) in enumerate(
        zip(target_keys, domain_keys, strict=True)
    ):
        target = _largest_key_source(target_child, target_key.name, facts)
        if target is None:
            continue

        for constraint_index, (
            target_constraint_key,
            domain_constraint_key,
        ) in enumerate(zip(target_keys, domain_keys, strict=True)):
            if constraint_index == filter_index:
                continue
            domain = _smallest_node_containing_all(
                domain_child,
                (domain_key.name, domain_constraint_key.name),
                facts,
            )
            if domain is None:
                continue
            if domain.rows / target.rows > threshold:
                continue
            constraint_domain = _smallest_key_producer(
                target_child,
                target_constraint_key.name,
                facts,
                require_selective=True,
                exclude=target.node,
            )
            if constraint_domain is None:
                continue
            if constraint_domain.rows / domain.rows > threshold:
                continue
            if contains_node(target.node, domain.node) or contains_node(
                target.node, constraint_domain.node
            ):
                continue
            if not domain_cost_is_small(domain, target, threshold):
                continue
            if not domain_cost_is_small(constraint_domain, domain, threshold):
                continue
            yield CompositeCandidate(
                target_side=target_side,
                target=target,
                target_key=target_key,
                domain=domain,
                domain_key=domain_key,
                constraint_domain=constraint_domain,
                domain_constraint_key=domain_constraint_key,
                target_constraint_key=target_constraint_key,
            )


def _make_domain(candidate: Candidate, ir: Join) -> IR:
    if isinstance(candidate, SimpleCandidate):
        return _project_bound_key(
            candidate.domain.node,
            candidate.domain.column,
            candidate.domain_key,
        )

    constraint_domain = _project_bound_key(
        candidate.constraint_domain.node,
        candidate.constraint_domain.column,
        candidate.target_constraint_key,
    )
    constrained = _make_semi_join(
        candidate.domain.node,
        expr.Col(
            candidate.domain.node.schema[candidate.domain.columns[1]],
            candidate.domain.columns[1],
        ),
        constraint_domain,
        expr.Col(
            constraint_domain.schema[candidate.target_constraint_key.name],
            candidate.target_constraint_key.name,
        ),
        nulls_equal=ir.options[1],
        suffix=ir.options[3],
    )
    return _project_bound_key(
        constrained, candidate.domain.column, candidate.domain_key
    )


def _project_bound_key(source: IR, bound_column: str, output_key: expr.Col) -> Select:
    """Project a bound source column under its join-visible key name."""
    dtype = source.schema[bound_column]
    assert dtype == output_key.dtype
    return Select(
        {output_key.name: dtype},
        (expr.NamedExpr(output_key.name, expr.Col(dtype, bound_column)),),
        True,  # noqa: FBT003
        source,
    )


def _make_semi_join(
    target: IR,
    target_key: expr.Col,
    domain: IR,
    domain_key: expr.Col,
    *,
    nulls_equal: bool,
    suffix: str,
) -> Join:
    return Join(
        target.schema,
        (expr.NamedExpr(target_key.name, target_key),),
        (expr.NamedExpr(domain_key.name, domain_key),),
        ("Semi", nulls_equal, None, suffix, False, "none"),
        target,
        domain,
    )


def make_producer(
    node: IR,
    columns: tuple[str, ...],
    path: tuple[int, ...],
    facts: PlanFacts,
) -> _Producer | None:
    """Construct a producer from gathered plan facts, if fully estimated."""
    rows = facts.row_estimates.get(node)
    source = facts.source_facts[node]
    if rows is None or rows <= 0 or source.cost is None:
        return None
    return _Producer(
        node=node,
        columns=columns,
        rows=rows,
        cost=source.cost,
        is_single_source=source.is_single_source,
        path=path,
    )


def _smallest_key_producer(
    root: IR,
    column: str,
    facts: PlanFacts,
    *,
    require_selective: bool,
    exclude: IR | None = None,
) -> _Producer | None:
    producers = []
    for reference, path in semijoin_pushdown_candidates(facts, root, column):
        node, bound_column = reference.node, reference.name
        if node is exclude:
            continue
        if require_selective and node not in facts.selective_nodes:
            continue
        producer = make_producer(node, (bound_column,), path, facts)
        if producer is not None:
            producers.append(producer)
    if not producers:
        return None
    return min(producers, key=lambda p: p.domain_score)


def _smallest_node_containing_all(
    root: IR, columns: Sequence[str], facts: PlanFacts
) -> _Producer | None:
    producers = []
    lineages: list[ColumnLineage] = []
    for column in columns:
        lineage = facts.column_lineages.get(ColumnRef(root, column))
        if lineage is None:
            return None
        lineages.append(lineage)
    if not lineages:
        return None
    path: tuple[int, ...] = ()
    while True:
        node = lineages[0].column.node
        if any(lineage.column.node != node for lineage in lineages[1:]):
            break
        bound_columns = tuple(lineage.column.name for lineage in lineages)
        producer = make_producer(node, bound_columns, path, facts)
        if producer is not None:
            producers.append(producer)
        if blocks_pushdown(node, facts):
            break
        source_child_index = lineages[0].source_child_index
        if source_child_index is None or any(
            lineage.source_child_index != source_child_index for lineage in lineages[1:]
        ):
            break
        sources = [lineage.source for lineage in lineages if lineage.source is not None]
        if len(sources) != len(lineages):
            # Some sources are None
            break
        path = (*path, source_child_index)
        lineages = sources
    if not producers:
        return None
    return min(producers, key=lambda p: p.domain_score)


def _largest_key_source(root: IR, column: str, facts: PlanFacts) -> _Producer | None:
    source_candidates = []
    fallback_candidates = []
    for reference, path in semijoin_pushdown_candidates(facts, root, column):
        node, bound_column = reference.node, reference.name
        producer = make_producer(node, (bound_column,), path, facts)
        if producer is None:
            continue
        item = (
            producer.rows,
            len(node.schema),
            producer,
        )
        if isinstance(node, (Scan, DataFrameScan)):
            source_candidates.append(item)
        else:
            fallback_candidates.append(item)
    candidates = source_candidates or fallback_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))[2]


def _estimate_join_rows(
    how: str, left_rows: int | None, right_rows: int | None
) -> int | None:
    if left_rows is None:
        return right_rows
    if right_rows is None:
        return left_rows
    if how in ("Inner", "Semi", "Anti"):
        return min(left_rows, right_rows)
    if how == "Left":
        return left_rows
    if how == "Right":
        return right_rows
    if how == "Full":
        return max(left_rows, right_rows)
    return None


def contains_node(root: IR, needle: IR) -> bool:
    """Return whether an equal node occurs in a DAG rooted at ``root``."""
    return needle in traversal([root])


def domain_cost_is_small(
    domain: _Producer, target: _Producer, threshold: float
) -> bool:
    """Return whether building a domain is cheap enough for its target."""
    return domain.cost / target.rows <= threshold


def has_filtering_semi_ancestor(root: IR, path: Sequence[int]) -> bool:
    """Return whether a selected child edge is below a filtering semi join."""
    node = root
    for child_index in path:
        if isinstance(node, Join) and node.options[0] == "Semi" and child_index == 0:
            return True
        node = node.children[child_index]
    return False


def _trace_decision(ir: Join, threshold: float, decision: Decision) -> None:
    join_filter_pushdown: dict[str, Any] = {
        "considered": True,
        "threshold": threshold,
        "reason": decision.reason,
    }
    record = {
        "scope": Scope.PLAN.value,
        "join_filter_pushdown": join_filter_pushdown,
        "actor_ir_id": ir.get_stable_id(),
        "actor_ir_type": type(ir).__name__,
    }
    if (candidate := decision.candidate) is not None:
        join_filter_pushdown.update(
            {
                "mode": candidate.mode,
                "target_side": candidate.target_side,
                "target_key": candidate.target_key.name,
                "domain_key": candidate.domain_key.name,
                "estimated_target_rows": candidate.target.rows,
                "estimated_domain_rows": candidate.domain.rows,
                "estimated_target_cost": candidate.target.cost,
                "estimated_domain_cost": candidate.domain.cost,
                "target_node_type": type(candidate.target.node).__name__,
                "domain_node_type": type(candidate.domain.node).__name__,
            }
        )
        if isinstance(candidate, CompositeCandidate):
            join_filter_pushdown.update(
                {
                    "constraint_key": candidate.target_constraint_key.name,
                    "estimated_constraint_rows": candidate.constraint_domain.rows,
                    "estimated_constraint_cost": candidate.constraint_domain.cost,
                }
            )
    log("Join Filter Pushdown", **record)


def _trace_aggregate_reuse(
    ir: GroupBy, candidate: _AggregateReuseCandidate | None
) -> None:
    aggregate_domain_reuse: dict[str, Any] = {
        "reason": "applied" if candidate is not None else "no_reusable_domain",
    }
    record = {
        "scope": Scope.PLAN.value,
        "join_aggregate_domain_reuse": aggregate_domain_reuse,
        "actor_ir_id": ir.get_stable_id(),
        "actor_ir_type": type(ir).__name__,
    }
    if candidate is not None:
        aggregate_domain_reuse.update(
            {
                "detail_side": candidate.detail_side,
                "value_column": candidate.value_column.name,
                "domain_node_type": candidate.domain_node_type,
                "estimated_replacement_rows": candidate.replacement_rows,
                "estimated_replacement_cost": candidate.replacement_cost,
            }
        )
    log("Join Aggregate Domain Reuse", **record)
