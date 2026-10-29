# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning and row-order metadata for the streaming actor graph."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    HashScheme,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
    RowOrdering,
    RowOrderingScope,
)
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Cast, Col, TemporalFunction
from cudf_polars.dsl.ir import HStack, Select
from cudf_polars.dsl.utils.column_domain import column_domain_bindings
from cudf_polars.dsl.utils.naming import indices_to_names, names_to_indices

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.channel_metadata import PartitioningSpecValue
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.expr import Expr, NamedExpr
    from cudf_polars.dsl.ir import IR


type InterRankScheme = HashScheme | OrderScheme | None
type PartitioningScheme = InterRankScheme | Literal["inherit"]
type PartitioningLevel = Literal["flat", "inter_rank", "local"]
type DirectColumnTargets = dict[str, tuple[str, ...]]


def _hash_keys_match(
    scheme: HashScheme, key_indices: tuple[int, ...], *, allow_subset: bool
) -> bool:
    current = scheme.column_indices
    target = key_indices[: len(current)] if allow_subset else key_indices
    return target == current


def _ordering_keys_match(
    ordering: Ordering,
    key_indices: tuple[int, ...],
    *,
    order_keys: tuple[OrderKey, ...] | None,
    allow_subset: bool,
) -> bool:
    n_keys = len(ordering.keys)
    if allow_subset:
        if n_keys > len(key_indices):
            return False
    elif n_keys != len(key_indices):
        return False
    else:
        n_keys = len(key_indices)
    if order_keys is not None:
        return ordering.keys == order_keys[:n_keys]
    return tuple(k.column_index for k in ordering.keys) == key_indices[:n_keys]


def _matching_order_scheme(
    scheme: OrderScheme,
    key_indices: tuple[int, ...],
    *,
    order_keys: tuple[OrderKey, ...] | None,
    allow_subset: bool,
) -> OrderScheme | None:
    matches = [
        (index, ordering)
        for index, ordering in enumerate(scheme.orderings)
        if _ordering_keys_match(
            ordering,
            key_indices,
            order_keys=order_keys,
            allow_subset=allow_subset,
        )
    ]
    if not matches:
        return None
    # Prefer the longest match; equal-length ties preserve metadata order.
    index, ordering = max(matches, key=lambda match: len(match[1].keys))
    return OrderScheme(
        (
            ordering,
            *scheme.orderings[:index],
            *scheme.orderings[index + 1 :],
        )
    )


def _keys_match(
    scheme: PartitioningSpecValue,
    key_indices: tuple[int, ...],
    *,
    order_keys: tuple[OrderKey, ...] | None,
    allow_subset: bool,
) -> InterRankScheme:
    if isinstance(scheme, HashScheme) and _hash_keys_match(
        scheme, key_indices, allow_subset=allow_subset
    ):
        return scheme
    if isinstance(scheme, OrderScheme):
        return _matching_order_scheme(
            scheme,
            key_indices,
            order_keys=order_keys,
            allow_subset=allow_subset,
        )
    return None


def _row_ordering_implies(actual: RowOrdering, required: RowOrdering) -> bool:
    """
    Return whether ``actual`` proves ``required``.

    Global ordering also holds within each rank. Refining the groups preserves
    ordering after newly grouped columns are removed from the order keys.
    """
    if (
        required.scope == RowOrderingScope.GLOBAL
        and actual.scope != RowOrderingScope.GLOBAL
    ):
        return False

    actual_groups = frozenset(actual.group_keys)
    required_groups = frozenset(required.group_keys)
    if not actual_groups.issubset(required_groups):
        return False

    newly_grouped = required_groups - actual_groups
    remaining_keys = tuple(
        key for key in actual.order_keys if key.column_index not in newly_grouped
    )
    return remaining_keys[: len(required.order_keys)] == required.order_keys


def is_row_ordered_by(
    partitioning: Partitioning | None,
    order_keys: Sequence[OrderKey],
    *,
    group_keys: Sequence[int] = (),
    scope: RowOrderingScope = RowOrderingScope.RANK_LOCAL,
) -> bool:
    """Return whether metadata proves the requested row ordering."""
    if partitioning is None or not order_keys:
        return False
    required = RowOrdering(order_keys, scope, group_keys=group_keys)
    return any(
        _row_ordering_implies(actual, required) for actual in partitioning.row_orderings
    )


def _deduplicate_row_orderings(
    orderings: Sequence[RowOrdering],
) -> tuple[RowOrdering, ...]:
    """Remove exact duplicate row-ordering facts while preserving order."""
    result: list[RowOrdering] = []
    for ordering in orderings:
        if ordering not in result:
            result.append(ordering)
    return tuple(result)


def partitioning_or_none(
    inter_rank: InterRankScheme,
    local: PartitioningScheme,
    row_orderings: Sequence[RowOrdering],
) -> Partitioning | None:
    """Build partitioning without duplicates, orphaned inheritance, or empty state."""
    if inter_rank is None and local == "inherit":
        local = None
    row_orderings = _deduplicate_row_orderings(row_orderings)
    if inter_rank is None and local is None and not row_orderings:
        return None
    return Partitioning(inter_rank, local, row_orderings=row_orderings)


def replace_row_orderings(
    partitioning: Partitioning,
    row_orderings: Sequence[RowOrdering],
) -> Partitioning:
    """Return partitioning with exactly the supplied row-ordering facts."""
    if not row_orderings:
        raise ValueError("At least one row ordering is required")
    result = partitioning_or_none(
        partitioning.inter_rank,
        partitioning.local,
        row_orderings,
    )
    assert result is not None
    return result


def add_row_ordering(
    partitioning: Partitioning,
    row_ordering: RowOrdering,
) -> Partitioning:
    """Add one row-ordering fact without removing existing facts."""
    return replace_row_orderings(
        partitioning,
        (*partitioning.row_orderings, row_ordering),
    )


def remap_ordering_keys(ordering: Ordering, new_indices: tuple[int, ...]) -> Ordering:
    """Return an ordering with keys remapped to another schema."""
    return ordering.with_keys(_remap_order_keys(ordering.keys, new_indices))


def _remap_order_keys(
    keys: Sequence[OrderKey], new_indices: Sequence[int]
) -> tuple[OrderKey, ...]:
    """Return order keys with new column indices."""
    return tuple(
        OrderKey(index, key.order, key.null_order)
        for key, index in zip(keys, new_indices, strict=True)
    )


def _unwrap_truncate_transparent_casts(expr: Expr) -> Expr:
    while isinstance(expr, Cast):
        src_id = expr.children[0].dtype.id()
        dst_id = expr.dtype.id()
        if src_id != dst_id and {
            src_id,
            dst_id,
        } != {plc.TypeId.INT64, plc.TypeId.TIMESTAMP_NANOSECONDS}:
            break
        (expr,) = expr.children
    return expr


def _truncate_source_name(expr: Expr) -> str | None:
    expr = _unwrap_truncate_transparent_casts(expr)
    if (
        isinstance(expr, TemporalFunction)
        and expr.name is TemporalFunction.Name.Truncate
    ):
        source = _unwrap_truncate_transparent_casts(expr.children[0])
        if isinstance(source, Col):
            return source.name
    return None


@dataclass(frozen=True)
class _DerivedKey:
    """A derived key that replaces an ordered-key suffix after a retained prefix."""

    retained_prefix_width: int
    order: plc.types.Order
    null_order: plc.types.NullOrder
    preserves_strictness: bool


def _match_derived_key(
    ne: NamedExpr,
    key_names: Sequence[str],
    keys: Sequence[OrderKey],
) -> _DerivedKey | None:
    """Match an expression that replaces an ordered-key suffix with one key."""
    source_name = _truncate_source_name(ne.value)
    if source_name is None:
        return None
    try:
        source_index = key_names.index(source_name)
    except ValueError:
        return None
    source_key = keys[source_index]
    return _DerivedKey(
        retained_prefix_width=source_index,
        order=source_key.order,
        null_order=source_key.null_order,
        preserves_strictness=False,
    )


def _direct_column_targets(ir: IR, child_index: int) -> DirectColumnTargets:
    """Map child columns to output columns that retain the same values."""
    targets: defaultdict[str, list[str]] = defaultdict(list)
    for output_name, binding in column_domain_bindings(ir).items():
        if binding.child_index == child_index:
            targets[binding.name].append(output_name)
    return {name: tuple(names) for name, names in targets.items()}


def _remap_direct_names(
    names: Sequence[str],
    targets: DirectColumnTargets,
    *,
    allow_prefix: bool = False,
) -> tuple[str, ...] | None:
    """Map names through direct columns, optionally retaining a valid prefix."""
    if not names:
        return ()
    remapped: list[str] = []
    for name in names:
        if name not in targets:
            if allow_prefix:
                break
            return None
        remapped.append(name if name in targets[name] else targets[name][0])
    return tuple(remapped) or None


@dataclass(frozen=True)
class _KeyRemapping:
    """One valid remapping of an ordered key sequence."""

    keys: tuple[OrderKey, ...]
    boundary_expr: NamedExpr | None = None
    boundary_prefix: int = 0
    preserves_strictness: bool = True


def _remap_key_candidates(
    ir: IR,
    child: IR,
    keys: Sequence[OrderKey],
    targets: DirectColumnTargets,
    expressions: Sequence[NamedExpr],
) -> list[_KeyRemapping]:
    """Return direct and derived output-key mappings for one key sequence."""
    key_names = indices_to_names(tuple(key.column_index for key in keys), child.schema)
    candidates: list[_KeyRemapping] = []

    target_names = _remap_direct_names(key_names, targets, allow_prefix=True)
    if target_names is not None:
        candidates.append(
            _KeyRemapping(
                _remap_order_keys(
                    keys[: len(target_names)],
                    names_to_indices(target_names, ir.schema),
                )
            )
        )
        if len(target_names) == 1:
            candidates.extend(
                _KeyRemapping(
                    _remap_order_keys(keys[:1], names_to_indices((alias,), ir.schema))
                )
                for alias in targets[key_names[0]]
                if alias != target_names[0]
            )

    for ne in expressions:
        derivation = _match_derived_key(ne, key_names, keys)
        if derivation is None:
            continue
        prefix_names = _remap_direct_names(
            key_names[: derivation.retained_prefix_width], targets
        )
        if prefix_names is None:
            continue
        output_names = (*prefix_names, ne.name)
        output_keys = (
            *keys[: derivation.retained_prefix_width],
            OrderKey(0, derivation.order, derivation.null_order),
        )
        candidates.append(
            _KeyRemapping(
                _remap_order_keys(
                    output_keys, names_to_indices(output_names, ir.schema)
                ),
                boundary_expr=ne,
                boundary_prefix=derivation.retained_prefix_width,
                preserves_strictness=derivation.preserves_strictness,
            )
        )
    return candidates


def _remap_row_orderings(
    ir: IR,
    child: IR,
    row_orderings: Sequence[RowOrdering],
    targets: DirectColumnTargets,
    expressions: Sequence[NamedExpr],
) -> tuple[RowOrdering, ...]:
    """Remap valid row-ordering facts to output columns."""
    remapped: list[RowOrdering] = []
    for ordering in row_orderings:
        group_names = indices_to_names(ordering.group_keys, child.schema)
        target_group_names = _remap_direct_names(group_names, targets)
        if target_group_names is None:
            continue
        group_indices = names_to_indices(target_group_names, ir.schema)
        remapped.extend(
            RowOrdering(
                candidate.keys,
                ordering.scope,
                group_keys=group_indices,
            )
            for candidate in _remap_key_candidates(
                ir,
                child,
                ordering.order_keys,
                targets,
                expressions,
            )
        )
    return _deduplicate_row_orderings(remapped)


def _remap_partitioning_scheme(
    ir: IR,
    child: IR,
    scheme: PartitioningScheme,
    targets: DirectColumnTargets,
    expressions: Sequence[NamedExpr],
    context: Context | None,
) -> PartitioningScheme:
    """Remap one partitioning scheme through a selected child edge."""
    if isinstance(scheme, HashScheme):
        old_key_names = indices_to_names(scheme.column_indices, child.schema)
        target_key_names = _remap_direct_names(old_key_names, targets)
        if target_key_names is None:
            return None
        return HashScheme(names_to_indices(target_key_names, ir.schema), scheme.modulus)
    if isinstance(scheme, OrderScheme):
        new_orderings: list[Ordering] = []
        for ordering in scheme.orderings:
            candidates = _remap_key_candidates(
                ir,
                child,
                ordering.keys,
                targets,
                expressions,
            )
            new_orderings.extend(
                remap_ordering_keys(
                    ordering,
                    tuple(key.column_index for key in candidate.keys),
                )
                for candidate in candidates
                if candidate.boundary_expr is None
                and len(candidate.keys) == len(ordering.keys)
            )
            if context is None:
                continue
            old_key_names = indices_to_names(ordering.column_indices, child.schema)
            for candidate in candidates:
                if candidate.boundary_expr is None:
                    continue
                br = context.br()
                boundary_chunk = ordering.get_boundaries(br)
                stream = boundary_chunk.stream
                boundary_df = DataFrame.from_table(
                    boundary_chunk.table_view(),
                    old_key_names,
                    [child.schema[name] for name in old_key_names],
                    stream,
                )
                column = candidate.boundary_expr.evaluate(boundary_df)
                boundary_table = plc.concatenate.concatenate(
                    [
                        plc.Table(
                            [
                                *boundary_chunk.table_view().columns()[
                                    : candidate.boundary_prefix
                                ],
                                column.obj,
                            ]
                        )
                    ],
                    stream=stream,
                    mr=br.device_mr,
                )
                boundaries = TableChunk.from_pylibcudf_table(
                    boundary_table,
                    stream,
                    exclusive_view=True,
                    br=br,
                )
                new_orderings.append(
                    Ordering(
                        candidate.keys,
                        boundaries,
                        strict_boundaries=(
                            ordering.strict_boundaries
                            and candidate.preserves_strictness
                        ),
                    )
                )
        if new_orderings:
            return OrderScheme(new_orderings)
        return None
    if scheme not in (None, "inherit"):  # pragma: no cover
        return None
    return scheme


def maybe_remap_partitioning(
    ir: IR,
    partitioning: Partitioning | None,
    *,
    child_index: int | None = None,
    context: Context | None = None,
    preserves_row_order: bool | None = None,
) -> Partitioning | None:
    """
    Remap partitioning metadata through one IR child edge.

    Unary nodes infer their only child and whether they preserve row order.
    Multi-input nodes must specify both. Partitioning schemes survive row
    reordering, while row-ordering facts do not. A context is required only
    when transformed boundary values must be evaluated.
    """
    if len(ir.children) == 1:
        child_index = 0 if child_index is None else child_index
        if preserves_row_order is None:
            preserves_row_order = ir.preserves_output_order
    elif child_index is None or preserves_row_order is None:
        raise ValueError(
            "Multi-input nodes must specify child_index and preserves_row_order"
        )
    assert child_index is not None
    assert preserves_row_order is not None
    if not 0 <= child_index < len(ir.children):
        raise ValueError(f"Invalid child index {child_index} for {type(ir).__name__}")
    if partitioning is None:
        return None

    child = ir.children[child_index]
    targets = _direct_column_targets(ir, child_index)
    expressions = (
        ir.exprs
        if isinstance(ir, Select)
        else ir.columns
        if isinstance(ir, HStack)
        else ()
    )
    inter_rank = _remap_partitioning_scheme(
        ir, child, partitioning.inter_rank, targets, expressions, context
    )
    local = _remap_partitioning_scheme(
        ir, child, partitioning.local, targets, expressions, context
    )
    row_orderings = (
        _remap_row_orderings(
            ir, child, partitioning.row_orderings, targets, expressions
        )
        if preserves_row_order
        else ()
    )
    return partitioning_or_none(inter_rank, local, row_orderings)


@dataclass(frozen=True)
class NormalizedPartitioning:
    """
    Partitioning schemes resolved for one requested key sequence.

    A scheme is ``None`` when metadata does not cover the requested keys.
    Local ``"inherit"`` means the local layout follows the inter-rank scheme.
    """

    inter_rank_scheme: InterRankScheme
    local_scheme: PartitioningScheme

    def __bool__(self) -> bool:
        """True when both inter-rank and local schemes are present."""
        return self.inter_rank_scheme is not None and self.local_scheme is not None

    def _scheme_for_level(self, level: PartitioningLevel) -> PartitioningScheme:
        """Return the scheme relevant to the requested partitioning level."""
        match level:
            case "flat":
                if self.local_scheme != "inherit":
                    return None
                return self.inter_rank_scheme
            case "inter_rank":
                return self.inter_rank_scheme
            case "local":
                return self.local_scheme

    @staticmethod
    def _scheme_is_strict(scheme: PartitioningScheme) -> bool:
        """True when one scheme proves strict partitioning."""
        if scheme is None or scheme == "inherit":
            return False
        if isinstance(scheme, OrderScheme):
            return scheme.orderings[0].strict_boundaries
        return True

    def is_strictly_partitioned(
        self,
        *,
        level: PartitioningLevel = "flat",
    ) -> bool:
        """True if data is strictly partitioned at the requested level."""
        return self._scheme_is_strict(self._scheme_for_level(level))

    def get_ordering(
        self,
        *,
        level: PartitioningLevel = "flat",
    ) -> Ordering | None:
        """Return a matching ordering for the requested partitioning level."""
        scheme = self._scheme_for_level(level)
        return scheme.orderings[0] if isinstance(scheme, OrderScheme) else None

    def is_aligned_with(
        self, other: NormalizedPartitioning, br: BufferResource
    ) -> bool:
        """True when both sides share identical inter-rank and local layouts."""

        def schemes_aligned(
            lhs: PartitioningScheme,
            rhs: PartitioningScheme,
        ) -> bool:
            if isinstance(lhs, OrderScheme):
                return isinstance(rhs, OrderScheme) and lhs.orderings[
                    0
                ].boundaries_aligned_with(rhs.orderings[0], br)
            if isinstance(lhs, HashScheme):
                return (
                    isinstance(rhs, HashScheme)
                    and lhs.modulus == rhs.modulus
                    and len(lhs.column_indices) == len(rhs.column_indices)
                )
            return lhs == "inherit" and rhs == "inherit"

        local_schemes_strict = (
            self.local_scheme == "inherit" and other.local_scheme == "inherit"
        ) or (
            self._scheme_is_strict(self.local_scheme)
            and self._scheme_is_strict(other.local_scheme)
        )
        return (
            self.is_strictly_partitioned(level="inter_rank")
            and other.is_strictly_partitioned(level="inter_rank")
            and local_schemes_strict
            and schemes_aligned(self.inter_rank_scheme, other.inter_rank_scheme)
            and schemes_aligned(self.local_scheme, other.local_scheme)
        )

    @classmethod
    def from_keys(
        cls,
        partitioning_metadata: Partitioning | None,
        nranks: int,
        *,
        keys: Sequence[int] | Sequence[OrderKey],
        allow_subset: bool = True,
    ) -> NormalizedPartitioning:
        """Resolve partitioning for column indices or fully specified order keys."""
        if not keys:
            return cls(None, None)
        if isinstance(keys[0], OrderKey):
            order_keys = tuple(cast("Sequence[OrderKey]", keys))
            key_indices = tuple(key.column_index for key in order_keys)
        else:
            order_keys = None
            key_indices = tuple(cast("Sequence[int]", keys))
        inter_rank_scheme, local_scheme = cls._normalize_schemes(
            partitioning_metadata,
            key_indices,
            nranks,
            order_keys=order_keys,
            allow_subset=allow_subset,
        )
        return cls(inter_rank_scheme, local_scheme)

    @staticmethod
    def _normalize_schemes(
        partitioning_metadata: Partitioning | None,
        key_indices: tuple[int, ...],
        nranks: int,
        *,
        order_keys: tuple[OrderKey, ...] | None,
        allow_subset: bool = False,
    ) -> tuple[InterRankScheme, PartitioningScheme]:
        """Translate metadata into schemes matching the requested keys."""
        if not key_indices:
            return None, None

        trivial = (
            HashScheme(key_indices, 1) if nranks == 1 and order_keys is None else None
        )
        if partitioning_metadata is None:
            return trivial, None

        matched_inter_rank = _keys_match(
            partitioning_metadata.inter_rank,
            key_indices,
            order_keys=order_keys,
            allow_subset=allow_subset,
        )
        inter_rank_scheme = matched_inter_rank or trivial
        if inter_rank_scheme is None and nranks > 1:
            return None, None

        matched_local = _keys_match(
            partitioning_metadata.local,
            key_indices,
            order_keys=order_keys,
            allow_subset=allow_subset,
        )
        local_scheme: PartitioningScheme
        if matched_local is not None:
            local_scheme = matched_local
        elif partitioning_metadata.local == "inherit":
            local_scheme = "inherit"
        else:
            local_scheme = None

        if matched_inter_rank is None and nranks == 1:
            if local_scheme not in (None, "inherit"):
                return local_scheme, "inherit"
            if trivial is not None:
                return trivial, None
            return None, None

        return inter_rank_scheme, local_scheme
