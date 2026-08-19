# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import DataFrameScan, Filter, GroupBy, Join, Select, Sort, Union
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.partitioning_hints import (
    HashPartitioningHint,
    NamedOrderKey,
    OrderPartitioningHint,
    collect_partitioning_hints,
)
from cudf_polars.utils.sorting import sort_order

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR

I64 = DataType(pl.Int64())
BOOL = DataType(pl.Boolean())


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


def test_sort_creates_order_partition_hint() -> None:
    scan = make_scan("a", "b")
    sort = make_sort(
        scan,
        "a",
        "b",
        descending=(False, True),
        nulls_last=(True, False),
    )

    hints = collect_partitioning_hints(
        sort,
        {sort: PartitionInfo(2), scan: PartitionInfo(2)},
    )

    assert hints[scan] == OrderPartitioningHint(
        (
            NamedOrderKey("a", descending=False, nulls_last=True),
            NamedOrderKey("b", descending=True, nulls_last=False),
        )
    )


def test_join_creates_hash_partitioning_hints() -> None:
    left = make_scan("k", "left_value")
    right = make_scan("k", "right_value")
    join = Join(
        {"k": I64, "left_value": I64, "right_value": I64},
        (named_col("k"),),
        (named_col("k"),),
        ("Inner", False, None, "_right", True, "none"),
        left,
        right,
    )

    hints = collect_partitioning_hints(
        join,
        {
            join: PartitionInfo(7),
            left: PartitionInfo(3),
            right: PartitionInfo(7),
        },
    )

    assert hints[left] == HashPartitioningHint(("k",), peer_partition_count=7)
    assert hints[right] == HashPartitioningHint(("k",), peer_partition_count=7)


def test_filter_clears_peer_partition_count() -> None:
    scan = make_scan("k", "value")
    mask = expr.NamedExpr("mask", expr.Literal(BOOL, True))  # noqa: FBT003
    filtered = Filter(scan.schema, mask, scan)
    right = make_scan("k", "right_value")
    join = Join(
        {"k": I64, "value": I64, "right_value": I64},
        (named_col("k"),),
        (named_col("k"),),
        ("Inner", False, None, "_right", True, "none"),
        filtered,
        right,
    )

    hints = collect_partitioning_hints(
        join,
        {
            join: PartitionInfo(7),
            filtered: PartitionInfo(3),
            scan: PartitionInfo(3),
            right: PartitionInfo(7),
        },
    )

    assert hints[filtered] == HashPartitioningHint(("k",), peer_partition_count=7)
    assert hints[scan] == HashPartitioningHint(("k",))
    assert hints[right] == HashPartitioningHint(("k",), peer_partition_count=7)


def test_groupby_preserves_peer_partition_count() -> None:
    scan = make_scan("k", "value")
    groupby = GroupBy(
        {"k": I64},
        (named_col("k"),),
        (),
        maintain_order=False,
        zlice=None,
        df=scan,
    )
    right = make_scan("k", "right_value")
    join = Join(
        {"k": I64, "right_value": I64},
        (named_col("k"),),
        (named_col("k"),),
        ("Inner", False, None, "_right", True, "none"),
        groupby,
        right,
    )

    hints = collect_partitioning_hints(
        join,
        {
            join: PartitionInfo(7),
            groupby: PartitionInfo(3),
            scan: PartitionInfo(3),
            right: PartitionInfo(7),
        },
    )

    assert hints[groupby] == HashPartitioningHint(("k",), peer_partition_count=7)
    assert hints[scan] == HashPartitioningHint(("k",), peer_partition_count=7)
    assert hints[right] == HashPartitioningHint(("k",), peer_partition_count=7)


def test_groupby_creates_hash_partition_hint() -> None:
    scan = make_scan("a", "b")
    groupby = GroupBy(
        {"a": I64},
        (named_col("a"),),
        (),
        maintain_order=False,
        zlice=None,
        df=scan,
    )

    hints = collect_partitioning_hints(
        groupby,
        {groupby: PartitionInfo(2), scan: PartitionInfo(2)},
    )

    assert hints[scan] == HashPartitioningHint(("a",))


def test_select_remaps_order_partition_hint() -> None:
    scan = make_scan("a", "b")
    select = Select(
        {"x": I64, "b": I64},
        (expr.NamedExpr("x", expr.Col(I64, "a")), named_col("b")),
        should_broadcast=False,
        df=scan,
    )
    sort = make_sort(select, "x")

    hints = collect_partitioning_hints(
        sort,
        {
            sort: PartitionInfo(2),
            select: PartitionInfo(2),
            scan: PartitionInfo(2),
        },
    )

    assert hints[scan] == OrderPartitioningHint(
        (NamedOrderKey("a", descending=False, nulls_last=False),)
    )


def test_groupby_remaps_order_partition_hint() -> None:
    scan = make_scan("a", "b")
    groupby = GroupBy(
        {"key": I64},
        (expr.NamedExpr("key", expr.Col(I64, "a")),),
        (),
        maintain_order=False,
        zlice=None,
        df=scan,
    )
    sort = make_sort(groupby, "key")

    hints = collect_partitioning_hints(
        sort,
        {
            sort: PartitionInfo(2),
            groupby: PartitionInfo(2),
            scan: PartitionInfo(2),
        },
    )

    assert hints[scan] == OrderPartitioningHint(
        (NamedOrderKey("a", descending=False, nulls_last=False),),
        strict_key_count=1,
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

    hints = collect_partitioning_hints(
        root,
        {root: PartitionInfo(2), scan: PartitionInfo(2)},
    )

    assert hints[scan] == OrderPartitioningHint(
        (
            NamedOrderKey("a", descending=False, nulls_last=False),
            NamedOrderKey("b", descending=False, nulls_last=False),
        )
    )


def test_fanout_merges_compatible_hash_hint_into_order_hint() -> None:
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

    hints = collect_partitioning_hints(
        root,
        {
            root: PartitionInfo(5),
            join: PartitionInfo(5),
            scan: PartitionInfo(3),
            right: PartitionInfo(5),
        },
    )

    assert hints[scan] == OrderPartitioningHint(
        (
            NamedOrderKey("a", descending=False, nulls_last=False),
            NamedOrderKey("b", descending=False, nulls_last=False),
        ),
        strict_key_count=1,
        peer_partition_count=5,
    )
    assert hints[right] == HashPartitioningHint(("a",), peer_partition_count=5)


def test_conflicting_fanout_drops_hint() -> None:
    scan = make_scan("a", "b")
    root = Union(
        scan.schema,
        None,
        False,  # noqa: FBT003
        make_sort(scan, "a"),
        make_sort(scan, "b"),
    )

    hints = collect_partitioning_hints(
        root,
        {root: PartitionInfo(2), scan: PartitionInfo(2)},
    )

    assert scan not in hints
