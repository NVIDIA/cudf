# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    DataFrameScan,
    GroupBy,
    Join,
    MapFunction,
    Select,
    Sort,
    Union,
)
from cudf_polars.streaming.partitioning_hints import (
    NamedOrderKey,
    OrderPartitioningHint,
    StrictPartitioningHint,
    collect_partitioning_hints,
)
from cudf_polars.utils.sorting import sort_order

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR

I64 = DataType(pl.Int64())


def make_scan(*names: str) -> DataFrameScan:
    frame = pl.DataFrame({name: [1] for name in names})
    return DataFrameScan(dict.fromkeys(names, I64), frame._df, None)


def named_col(name: str) -> expr.NamedExpr:
    return expr.NamedExpr(name, expr.Col(I64, name))


def make_sort(
    child: IR,
    *names: str,
    descending: tuple[bool, ...] | None = None,
    nulls_last: tuple[bool, ...] | None = None,
) -> Sort:
    if descending is None:
        descending = (False,) * len(names)
    if nulls_last is None:
        nulls_last = (False,) * len(names)
    order, null_order = sort_order(
        descending, nulls_last=nulls_last, num_keys=len(names)
    )
    return Sort(
        child.schema,
        tuple(named_col(name) for name in names),
        order,
        null_order,
        stable=False,
        zlice=None,
        df=child,
    )


def make_hint_sorted(
    child: IR,
    *names: str,
    descending: tuple[bool, ...] | None = None,
    nulls_last: tuple[bool, ...] | None = None,
) -> MapFunction:
    if descending is None:
        descending = (False,) * len(names)
    if nulls_last is None:
        nulls_last = (False,) * len(names)
    return MapFunction(
        child.schema, "hint_sorted", (names, descending, nulls_last), child
    )


def make_groupby(child: IR, *names: str) -> GroupBy:
    return GroupBy(
        dict.fromkeys(names, I64),
        tuple(named_col(name) for name in names),
        (),
        maintain_order=False,
        zlice=None,
        df=child,
    )


def test_sort_creates_order_partition_hint() -> None:
    scan = make_scan("a", "b")
    sort = make_sort(
        scan,
        "a",
        "b",
        descending=(False, True),
        nulls_last=(True, False),
    )

    hints = collect_partitioning_hints(sort)

    assert hints[scan] == (
        OrderPartitioningHint(
            (
                NamedOrderKey("a", descending=False, nulls_last=True),
                NamedOrderKey("b", descending=True, nulls_last=False),
            )
        ),
    )


def test_select_remaps_order_partition_hint() -> None:
    scan = make_scan("a", "b")
    select = Select(
        {"x": I64, "b": I64},
        (expr.NamedExpr("x", expr.Col(I64, "a")), named_col("b")),
        should_broadcast=False,
        df=scan,
    )
    sort = make_sort(select, "x")

    hints = collect_partitioning_hints(sort)

    assert hints[scan] == (
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),)
        ),
    )


def test_non_column_sort_does_not_create_hint() -> None:
    scan = make_scan("a")
    order, null_order = sort_order((False,), nulls_last=(False,), num_keys=1)
    sort = Sort(
        scan.schema,
        (expr.NamedExpr("literal", expr.Literal(I64, 1)),),
        order,
        null_order,
        stable=False,
        zlice=None,
        df=scan,
    )

    hints = collect_partitioning_hints(sort)

    assert hints == {}


def test_hint_sorted_creates_order_partition_hint() -> None:
    scan = make_scan("a", "b")
    hint_sorted = make_hint_sorted(scan, "a", descending=(True,))

    hints = collect_partitioning_hints(hint_sorted)

    assert hints[scan] == (
        OrderPartitioningHint((NamedOrderKey("a", descending=True, nulls_last=False),)),
    )


def test_select_remaps_strict_partition_hint() -> None:
    scan = make_scan("a")
    select = Select(
        {"x": I64},
        (expr.NamedExpr("x", expr.Col(I64, "a")),),
        should_broadcast=False,
        df=scan,
    )
    right = make_scan("x")
    join = Join(
        {"x": I64},
        (named_col("x"),),
        (named_col("x"),),
        ("Inner", False, None, "_right", True, "none"),
        select,
        right,
    )

    hints = collect_partitioning_hints(join)

    assert hints[scan] == (StrictPartitioningHint(("a",)),)
    assert hints[right] == (StrictPartitioningHint(("x",)),)


def test_select_drops_order_hint_on_non_column_output() -> None:
    scan = make_scan("a")
    select = Select(
        {"x": I64},
        (expr.NamedExpr("x", expr.Literal(I64, 1)),),
        should_broadcast=False,
        df=scan,
    )
    sort = make_sort(select, "x")

    hints = collect_partitioning_hints(sort)

    assert hints[select] == (
        OrderPartitioningHint(
            (NamedOrderKey("x", descending=False, nulls_last=False),)
        ),
    )
    assert scan not in hints


def test_select_drops_strict_hint_on_non_column_output() -> None:
    scan = make_scan("a")
    select = Select(
        {"x": I64},
        (expr.NamedExpr("x", expr.Literal(I64, 1)),),
        should_broadcast=False,
        df=scan,
    )
    right = make_scan("x")
    join = Join(
        {"x": I64},
        (named_col("x"),),
        (named_col("x"),),
        ("Inner", False, None, "_right", True, "none"),
        select,
        right,
    )

    hints = collect_partitioning_hints(join)

    assert hints[select] == (StrictPartitioningHint(("x",)),)
    assert hints[right] == (StrictPartitioningHint(("x",)),)
    assert scan not in hints


def test_hint_sorted_keeps_declared_order_with_compatible_downstream_sort() -> None:
    scan = make_scan("a", "b")
    hint_sorted = make_hint_sorted(scan, "a")
    sort = make_sort(hint_sorted, "a")

    hints = collect_partitioning_hints(sort)

    assert hints[scan] == (
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),)
        ),
    )


def test_hint_sorted_keeps_declared_order_with_extended_downstream_sort() -> None:
    scan = make_scan("a", "b")
    hint_sorted = make_hint_sorted(scan, "a")
    sort = make_sort(hint_sorted, "a", "b")

    hints = collect_partitioning_hints(sort)

    assert hints[scan] == (
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),)
        ),
    )


def test_hint_sorted_keeps_declared_order_with_incompatible_downstream_sort() -> None:
    scan = make_scan("a", "b")
    hint_sorted = make_hint_sorted(scan, "a", descending=(True,))
    sort = make_sort(hint_sorted, "a")

    hints = collect_partitioning_hints(sort)

    assert hints[scan] == (
        OrderPartitioningHint((NamedOrderKey("a", descending=True, nulls_last=False),)),
    )


def test_groupby_remaps_order_partition_hint() -> None:
    scan = make_scan("a", "b")
    groupby = make_groupby(
        Select(
            {"key": I64},
            (expr.NamedExpr("key", expr.Col(I64, "a")),),
            should_broadcast=False,
            df=scan,
        ),
        "key",
    )
    sort = make_sort(groupby, "key")

    hints = collect_partitioning_hints(sort)

    assert hints[scan] == (
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),),
            strict_key_count=1,
        ),
    )


def test_fanout_keeps_more_specific_compatible_order_hint() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_sort(scan, "a"),
        make_sort(scan, "a", "b"),
    )

    hints = collect_partitioning_hints(root)

    assert hints[scan] == (
        OrderPartitioningHint(
            (
                NamedOrderKey("a", descending=False, nulls_last=False),
                NamedOrderKey("b", descending=False, nulls_last=False),
            )
        ),
    )


def test_fanout_marks_compatible_order_hint_as_strict() -> None:
    scan = make_scan("a", "b")
    right = make_scan("a", "right_value")
    join = Join(
        {"a": I64, "b": I64, "right_value": I64},
        (named_col("a"),),
        (named_col("a"),),
        ("Inner", False, None, "_right", True, "none"),
        scan,
        right,
    )
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_sort(scan, "a", "b"),
        join,
    )

    hints = collect_partitioning_hints(root)

    assert hints[scan] == (
        OrderPartitioningHint(
            (
                NamedOrderKey("a", descending=False, nulls_last=False),
                NamedOrderKey("b", descending=False, nulls_last=False),
            ),
            strict_key_count=1,
        ),
    )
    assert hints[right] == (StrictPartitioningHint(("a",)),)


def test_fanout_merges_compatible_strict_hints() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_groupby(scan, "a"),
        make_groupby(scan, "a", "b"),
    )

    hints = collect_partitioning_hints(root)

    assert hints[scan] == (StrictPartitioningHint(("a",)),)


def test_fanout_keeps_incompatible_strict_candidates() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_groupby(scan, "a"),
        make_groupby(scan, "b"),
    )

    hints = collect_partitioning_hints(root)

    assert set(hints[scan]) == {
        StrictPartitioningHint(("a",)),
        StrictPartitioningHint(("b",)),
    }


def test_fanout_keeps_incompatible_order_and_strict_candidates() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_sort(scan, "a"),
        make_groupby(scan, "b"),
    )

    hints = collect_partitioning_hints(root)

    assert set(hints[scan]) == {
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),)
        ),
        StrictPartitioningHint(("b",)),
    }


def test_compatible_hint_merging_is_not_directional() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_groupby(scan, "a", "b"),
        make_groupby(scan, "a"),
    )
    assert collect_partitioning_hints(root)[scan] == (StrictPartitioningHint(("a",)),)

    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_groupby(scan, "b"),
        make_sort(scan, "a"),
    )
    assert set(collect_partitioning_hints(root)[scan]) == {
        StrictPartitioningHint(("b",)),
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),)
        ),
    }

    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_sort(scan, "a", "b"),
        make_sort(scan, "a"),
    )
    assert collect_partitioning_hints(root)[scan] == (
        OrderPartitioningHint(
            (
                NamedOrderKey("a", descending=False, nulls_last=False),
                NamedOrderKey("b", descending=False, nulls_last=False),
            )
        ),
    )


def test_conflicting_fanout_keeps_candidate_hints() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_sort(scan, "a"),
        make_sort(scan, "b"),
    )

    hints = collect_partitioning_hints(root)

    assert set(hints[scan]) == {
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),)
        ),
        OrderPartitioningHint(
            (NamedOrderKey("b", descending=False, nulls_last=False),)
        ),
    }


def test_repeated_fanout_candidate_is_merged() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_sort(scan, "a"),
        make_sort(scan, "b"),
        make_sort(scan, "a"),
    )

    hints = collect_partitioning_hints(root)

    assert len(hints[scan]) == 2
    assert set(hints[scan]) == {
        OrderPartitioningHint(
            (NamedOrderKey("a", descending=False, nulls_last=False),)
        ),
        OrderPartitioningHint(
            (NamedOrderKey("b", descending=False, nulls_last=False),)
        ),
    }
