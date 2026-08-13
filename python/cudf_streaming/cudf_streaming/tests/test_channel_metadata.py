# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming metadata types (Partitioning and ChannelMetadata)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
import pytest

from cudf_streaming import (
    ChannelMetadata,
    HashScheme,
    Ordering,
    OrderKey,
    OrderScheme,
    Partitioning,
    TableChunk,
)
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context


def _make_boundaries(context: Context, table: plc.Table) -> TableChunk:
    stream = context.br().stream_pool.get_stream()
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=False,
        br=context.br(),
    )


def _two_key_order_scheme(
    context: Context, *, strict_boundaries: bool = False
) -> OrderScheme:
    """Two-key OrderScheme with a 1-row boundary table (2 partitions)."""
    boundaries = _make_boundaries(
        context,
        plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [100], plc.DataType(plc.TypeId.INT64)
                ),
                plc.Column.from_iterable_of_py(
                    ["abc"], plc.DataType(plc.TypeId.STRING)
                ),
            ]
        ),
    )
    return OrderScheme(
        [
            Ordering(
                [
                    OrderKey(
                        0,
                        plc.types.Order.ASCENDING,
                        plc.types.NullOrder.BEFORE,
                    ),
                    OrderKey(
                        1,
                        plc.types.Order.DESCENDING,
                        plc.types.NullOrder.AFTER,
                    ),
                ],
                boundaries,
                strict_boundaries=strict_boundaries,
            )
        ]
    )


def test_hash_scheme() -> None:
    """Test HashScheme construction, properties, equality, and repr."""
    h1 = HashScheme((0, 1), 16)
    assert h1.column_indices == (0, 1)
    assert h1.modulus == 16
    assert repr(h1) == "HashScheme((0, 1), 16)"

    # Equality
    assert h1 == HashScheme((0, 1), 16)
    assert h1 != HashScheme((0, 1), 32)
    assert h1 != HashScheme((2,), 16)


def test_order_key() -> None:
    k = OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    assert k.column_index == 0
    assert k.order == plc.types.Order.ASCENDING
    assert k.null_order == plc.types.NullOrder.BEFORE
    assert k == OrderKey(
        0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    )
    assert k != OrderKey(
        1, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    )
    assert "OrderKey" in repr(k)


def test_order_scheme(context: Context) -> None:
    """Test OrderScheme construction, properties, equality, and repr."""
    o1 = _two_key_order_scheme(context)
    ordering = o1.orderings[0]
    assert ordering.keys == (
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    )
    assert ordering.column_indices == (0, 1)
    assert not ordering.strict_boundaries
    assert ordering.num_boundaries == 1
    assert "OrderScheme" in repr(o1)

    assert o1.orderings[0].boundaries_aligned_with(
        _two_key_order_scheme(context).orderings[0], context.br()
    )

    o_strict = _two_key_order_scheme(context, strict_boundaries=True)
    assert o_strict.orderings[0].strict_boundaries
    assert not o1.orderings[0].boundaries_aligned_with(
        o_strict.orderings[0], context.br()
    )
    assert o_strict.orderings[0].boundaries_aligned_with(
        _two_key_order_scheme(context, strict_boundaries=True).orderings[0],
        context.br(),
    )

    with pytest.raises(TypeError, match="OrderKey"):
        OrderScheme(
            [
                Ordering(
                    [
                        (
                            0,
                            plc.types.Order.ASCENDING,
                            plc.types.NullOrder.BEFORE,
                        )
                    ],  # ty: ignore[invalid-argument-type]
                    _make_boundaries(
                        context,
                        plc.Table(
                            [
                                plc.Column.from_iterable_of_py(
                                    [0], plc.DataType(plc.TypeId.INT64)
                                )
                            ]
                        ),
                    ),
                )
            ],  # ty: ignore[invalid-argument-type]
        )

    with pytest.raises(ValueError, match="empty"):
        OrderScheme([])


def test_order_scheme_multiple_orderings(context: Context) -> None:
    """OrderScheme stores orderings valid for the same stream."""
    first = Ordering(
        [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)],
        _make_boundaries(
            context,
            plc.Table(
                [
                    plc.Column.from_iterable_of_py(
                        [100], plc.DataType(plc.TypeId.INT64)
                    )
                ]
            ),
        ),
        strict_boundaries=True,
    )
    second = Ordering(
        [OrderKey(2, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER)],
        _make_boundaries(
            context,
            plc.Table(
                [
                    plc.Column.from_iterable_of_py(
                        [200], plc.DataType(plc.TypeId.INT64)
                    )
                ]
            ),
        ),
    )
    scheme = OrderScheme([first, second])

    assert len(scheme.orderings) == 2
    assert scheme.orderings[0].keys == first.keys
    assert scheme.orderings[0].strict_boundaries == first.strict_boundaries
    assert scheme.orderings[0].num_boundaries == first.num_boundaries
    assert scheme.orderings[1].keys == second.keys


def test_order_scheme_get_boundaries(context: Context) -> None:
    scheme = _two_key_order_scheme(context)
    ordering = scheme.orderings[0]
    chunk = ordering.get_boundaries(context.br())
    assert chunk.table_view().num_columns() == 2
    assert chunk.table_view().num_rows() == 1
    scheme2 = OrderScheme(
        [
            Ordering(
                ordering.keys,
                chunk,
                strict_boundaries=ordering.strict_boundaries,
            )
        ]
    )
    assert scheme2.orderings[0].boundaries_aligned_with(
        scheme.orderings[0], context.br()
    )


def test_ordering_with_keys(context: Context) -> None:
    """with_keys shares boundaries and updates column indices."""
    o1 = _two_key_order_scheme(context)
    ordering = o1.orderings[0]
    new_keys = [
        OrderKey(5, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(3, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    ]
    ordering2 = ordering.with_keys(new_keys)
    assert ordering2.keys[0].column_index == 5
    assert ordering2.keys[1].column_index == 3
    assert ordering2.num_boundaries == ordering.num_boundaries
    assert ordering2.strict_boundaries == ordering.strict_boundaries
    # Schemes with different key indices but shared boundaries are boundary-aligned
    assert ordering.boundaries_aligned_with(ordering2, context.br())


def test_ordering_boundaries_aligned_with(context: Context) -> None:
    """Boundary comparison ignores key indices but checks values and ordering."""
    df = plc.Table(
        [
            plc.Column.from_iterable_of_py(
                [100, 200], plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_iterable_of_py(
                ["abc", "xyz"], plc.DataType(plc.TypeId.STRING)
            ),
        ]
    )
    keys = [
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    ]
    o1 = Ordering(keys, _make_boundaries(context, df))
    o2 = Ordering(keys, _make_boundaries(context, df))
    assert o1.boundaries_aligned_with(o2, context.br())

    # Different key column indices but same boundary values → still aligned
    shifted_keys = [
        OrderKey(2, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(3, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    ]
    o_shifted = Ordering(shifted_keys, _make_boundaries(context, df))
    assert o1.boundaries_aligned_with(o_shifted, context.br())

    # Different boundary values → not aligned (shape matches, values differ)
    df_diff = plc.Table(
        [
            plc.Column.from_iterable_of_py(
                [100, 300], plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_iterable_of_py(
                ["abc", "xyz"], plc.DataType(plc.TypeId.STRING)
            ),
        ]
    )
    o3 = Ordering(keys, _make_boundaries(context, df_diff))
    assert not o1.boundaries_aligned_with(o3, context.br())

    # Different strict_boundaries → not aligned
    o_strict = Ordering(
        keys,
        _make_boundaries(context, df),
        strict_boundaries=True,
    )
    assert not o1.boundaries_aligned_with(o_strict, context.br())


def test_order_scheme_key_column_mismatch(context: Context) -> None:
    """OrderScheme rejects key/column count mismatch."""
    boundaries = _make_boundaries(
        context,
        plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [0], plc.DataType(plc.TypeId.INT64)
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="keys must match"):
        OrderScheme(
            [
                Ordering(
                    [
                        OrderKey(
                            0,
                            plc.types.Order.ASCENDING,
                            plc.types.NullOrder.BEFORE,
                        ),
                        OrderKey(
                            1,
                            plc.types.Order.DESCENDING,
                            plc.types.NullOrder.AFTER,
                        ),
                    ],
                    boundaries,  # 1 column, but 2 keys
                )
            ]
        )


def test_partitioning_scenarios(context: Context) -> None:
    """Test various partitioning configurations."""
    # Default / None
    p_default = Partitioning()
    assert p_default.inter_rank is None
    assert p_default.local is None
    assert Partitioning(None, None).inter_rank is None
    assert Partitioning(None, None).local is None

    # Direct global shuffle: inter_rank=Hash, local=Aligned
    p_global = Partitioning(HashScheme((0,), 16), "inherit")
    assert p_global.inter_rank == HashScheme((0,), 16)
    assert p_global.local == "inherit"

    # Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    p_twostage = Partitioning(HashScheme((0,), 4), HashScheme((0,), 8))
    assert p_twostage.inter_rank == HashScheme((0,), 4)
    assert p_twostage.local == HashScheme((0,), 8)

    # Order-based partitioning (range partitioned / sorted)
    order_scheme = _two_key_order_scheme(context)
    p_ordered = Partitioning(order_scheme, "inherit")
    assert isinstance(p_ordered.inter_rank, OrderScheme)
    assert p_ordered.inter_rank.orderings[0].boundaries_aligned_with(
        order_scheme.orderings[0], context.br()
    )
    assert p_ordered.local == "inherit"

    # Mixed: inter_rank=Order, local=Hash
    p_mixed = Partitioning(
        _two_key_order_scheme(context),
        HashScheme((1,), 8),
    )
    assert isinstance(p_mixed.inter_rank, OrderScheme)
    assert isinstance(p_mixed.local, HashScheme)

    # Repr
    assert "Partitioning" in repr(p_global)
    assert "inter_rank" in repr(p_global)

    # Invalid type
    with pytest.raises(TypeError):
        Partitioning("invalid", None)  # ty: ignore[invalid-argument-type]


def test_channel_metadata() -> None:
    """Test ChannelMetadata construction and properties."""
    # Basic construction
    m = ChannelMetadata(local_count=4)
    assert m.local_count == 4
    assert not m.duplicated

    # With partitioning and duplicated
    p = Partitioning(HashScheme((0,), 16), "inherit")
    m_full = ChannelMetadata(local_count=4, partitioning=p, duplicated=True)
    assert m_full.partitioning.inter_rank == HashScheme((0,), 16)
    assert m_full.partitioning.local == "inherit"
    assert m_full.duplicated

    # Field comparisons (ChannelMetadata.__eq__ removed)
    m2 = ChannelMetadata(local_count=4)
    assert m.local_count == m2.local_count
    assert m.duplicated == m2.duplicated
    assert ChannelMetadata(local_count=8).local_count != m.local_count
    assert "local_count=4" in repr(m)

    # Validation
    with pytest.raises(ValueError, match="local_count must be non-negative"):
        ChannelMetadata(local_count=-1)


def test_message_roundtrip() -> None:
    """Test ChannelMetadata can round-trip through Message."""
    m = ChannelMetadata(
        local_count=4,
        partitioning=Partitioning(HashScheme((0,), 16), "inherit"),
        duplicated=True,
    )
    msg_m = Message(99, m)
    assert msg_m.sequence_number == 99
    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 4
    assert got_m.duplicated
    assert got_m.partitioning.inter_rank == HashScheme((0,), 16)
    assert msg_m.empty()


def test_message_roundtrip_with_order_scheme(context: Context) -> None:
    """Test ChannelMetadata with OrderScheme can round-trip through Message."""
    table = plc.Table(
        [
            plc.Column.from_iterable_of_py(
                [100, 200], plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_iterable_of_py(
                ["abc", "xyz"], plc.DataType(plc.TypeId.STRING)
            ),
        ]
    )
    boundaries = _make_boundaries(context, table)
    order_scheme = OrderScheme(
        [
            Ordering(
                [
                    OrderKey(
                        0,
                        plc.types.Order.ASCENDING,
                        plc.types.NullOrder.BEFORE,
                    ),
                    OrderKey(
                        1,
                        plc.types.Order.DESCENDING,
                        plc.types.NullOrder.AFTER,
                    ),
                ],
                boundaries,
                strict_boundaries=True,
            )
        ]
    )
    m = ChannelMetadata(
        local_count=8,
        partitioning=Partitioning(order_scheme, "inherit"),
        duplicated=True,
    )
    msg_m = Message(42, m)
    assert msg_m.sequence_number == 42
    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 8
    assert got_m.duplicated
    assert isinstance(got_m.partitioning.inter_rank, OrderScheme)
    ordering = got_m.partitioning.inter_rank.orderings[0]
    assert ordering.keys == (
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    )
    assert got_m.partitioning.local == "inherit"
    assert ordering.strict_boundaries
    assert ordering.num_boundaries == 2
    assert got_m.partitioning.inter_rank.orderings[0].boundaries_aligned_with(
        order_scheme.orderings[0], context.br()
    )
    assert msg_m.empty()


def test_order_scheme_roundtrip_from_metadata(context: Context) -> None:
    """An OrderScheme read back from ChannelMetadata can be re-used in a new Partitioning."""
    src = ChannelMetadata(
        local_count=1,
        partitioning=Partitioning(_two_key_order_scheme(context), "inherit"),
    )
    scheme = src.partitioning.inter_rank
    assert isinstance(scheme, OrderScheme)

    p2 = Partitioning(scheme, None)
    assert isinstance(p2.inter_rank, OrderScheme)
    assert p2.inter_rank.orderings[0].boundaries_aligned_with(
        _two_key_order_scheme(context).orderings[0], context.br()
    )


def test_access_after_move_raises() -> None:
    """Test that accessing a released ChannelMetadata raises ValueError."""
    m = ChannelMetadata(
        local_count=4,
        partitioning=Partitioning(HashScheme((0,), 16), "inherit"),
    )
    # Move into a message (releases the handle)
    _ = Message(0, m)

    # Accessing any property should raise ValueError
    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.local_count

    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.partitioning

    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.duplicated

    with pytest.raises(ValueError, match="uninitialized"):
        repr(m)
