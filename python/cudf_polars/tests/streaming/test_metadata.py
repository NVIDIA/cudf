# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF metadata functionality."""

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk

from cudf_polars import Translator
from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import GroupBy, HStack, Projection, Select, Sort
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.collectives.sort import (
    _sort_to_order_keys,
)
from cudf_polars.streaming.actor_graph.core import evaluate_logical_plan
from cudf_polars.streaming.actor_graph.utils import (
    NormalizedPartitioning,
    maybe_remap_partitioning,
)
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture(scope="module")
def left() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )


@pytest.fixture(scope="module")
def right() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(
            max_rows_per_partition=1,
            broadcast_limit=48,
            dynamic_planning=None,
        ),
        StreamingOptions(
            max_rows_per_partition=1,
            broadcast_limit=240,
            dynamic_planning=None,
        ),
    ],
)
def test_rapidsmpf_join_metadata(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    spmd_engine_factory,
    options,
) -> None:
    # Pinned to SPMD: ``ChannelMetadata.__reduce_cython__`` can't pickle
    # ``self._handle`` across worker/actor processes, so the
    # ``metadata_collector`` round-trip fails on Dask and Ray.
    #
    # When https://github.com/rapidsai/cudf/pull/22394 lands, dedup of
    # replicated outputs moves to the Dask/Ray frontends and the
    # ``duplicated`` flag's semantics change to "every rank holds the
    # data". Revisit the ``len(metadata_collector) == 1`` and
    # ``metadata.duplicated is False`` assertions below, and reconsider
    # whether this test can widen to ``streaming_engine_factory``.
    engine = spmd_engine_factory(options)
    config_options = ConfigOptions.from_polars_engine(engine)
    broadcast_join_limit = (
        config_options.executor.broadcast_limit
        // config_options.executor.target_partition_size
    )
    q = left.join(
        right,
        on="y",
        how="left",
    ).filter(pl.col("x") > pl.col("zz"))
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    left_count = left.collect(engine=engine).height
    right_count = right.collect(engine=engine).height

    metadata_collector = evaluate_logical_plan(
        ir, config_options, collect_metadata=True
    )[1]
    assert metadata_collector is not None
    assert len(metadata_collector) == 1
    metadata = metadata_collector[0]
    assert metadata.local_count == left_count
    assert metadata.duplicated is False
    if right_count > broadcast_join_limit:
        # After shuffle, partitioning has inter_rank=HashScheme, local="inherit"
        assert isinstance(metadata.partitioning.inter_rank, HashScheme)
        # "y" is at index 1 in the output schema: ["x", "y", "z", "xx", "zz"]
        assert metadata.partitioning.inter_rank.column_indices == (1,)
        assert metadata.partitioning.local == "inherit"
    else:
        # No partitioning (broadcast join preserves no partitioning from IO)
        assert metadata.partitioning.inter_rank is None
        assert metadata.partitioning.local is None


@pytest.mark.parametrize(
    "partitioning,key_indices,nranks,expected",
    [
        (
            None,
            (0, 1),
            1,
            NormalizedPartitioning(HashScheme((0, 1), 1), None),
        ),
        (None, (0, 1), 4, NormalizedPartitioning(None, None)),
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
        ),
        (
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), HashScheme((0, 1), 4)),
        ),
        (
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=HashScheme((0,), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), None),
        ),
        (
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        (
            Partitioning(inter_rank=HashScheme((1, 0), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        (
            Partitioning(
                inter_rank=None,
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            1,
            NormalizedPartitioning(HashScheme((0, 1), 4), "inherit"),
        ),
        (
            Partitioning(
                inter_rank=None,
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        (
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=None,
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), None),
        ),
    ],
)
def test_get_partitioning_moduli(partitioning, key_indices, nranks, expected) -> None:
    """from_keys(..., allow_subset=False) matches expected NormalizedPartitioning."""
    state = NormalizedPartitioning.from_keys(
        partitioning, nranks, keys=key_indices, allow_subset=False
    )
    assert state == expected


@pytest.mark.parametrize(
    "partitioning,key_indices,nranks,expected",
    [
        # Partitioned on (0,); keys (0, 1) → prefix (0,) matches
        (
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
        ),
        # Partitioned on (0, 1); keys (0, 1, 2) → prefix (0, 1) matches
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1, 2),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
        ),
        # Partitioned on (0,) with explicit local; keys (0, 1) → prefix matches
        (
            Partitioning(
                inter_rank=HashScheme((0,), 8),
                local=HashScheme((0,), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
        ),
        # Full key match with allow_subset: same as exact match
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
        ),
        # Keys (0,) are shorter than partition (0, 1) → no prefix match
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0,),
            4,
            NormalizedPartitioning(None, None),
        ),
        # Partitioned on (1,); keys (0, 1) → prefix of keys is (0,), not (1,) → no match
        (
            Partitioning(inter_rank=HashScheme((1,), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        # Resolves https://github.com/rapidsai/cudf/issues/21742
        (
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (1,),
            1,
            NormalizedPartitioning(HashScheme((1,), 1), None),
        ),
    ],
)
def test_get_partitioning_moduli_allow_subset(
    partitioning, key_indices, nranks, expected
) -> None:
    """from_keys(..., allow_subset=True) matches expected NormalizedPartitioning."""
    state = NormalizedPartitioning.from_keys(
        partitioning, nranks, keys=key_indices, allow_subset=True
    )
    assert state == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        # Same inter_rank modulus and key count, both "inherit" → aligned
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            NormalizedPartitioning(HashScheme((2, 3), 8), "inherit"),
            True,
        ),
        # Same modulus, same local HashScheme arity → aligned (column_indices not compared)
        (
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
            NormalizedPartitioning(HashScheme((1,), 8), HashScheme((1,), 4)),
            True,
        ),
        # Different inter_rank modulus → not aligned
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 4), "inherit"),
            False,
        ),
        # Different key count → not aligned
        (
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            False,
        ),
        # One side has local=None → not aligned (bool is False)
        (
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0,), 8), None),
            False,
        ),
        # Both None inter_rank → not aligned
        (
            NormalizedPartitioning(None, None),
            NormalizedPartitioning(None, None),
            False,
        ),
        # Mismatched local types: "inherit" vs HashScheme → not aligned
        (
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
            False,
        ),
        # Local HashSchemes with different modulus → not aligned
        (
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 2)),
            False,
        ),
        # Inter-rank aligned; local HashSchemes same modulus but different arity → not aligned
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), HashScheme((0,), 4)),
            NormalizedPartitioning(HashScheme((0, 1), 8), HashScheme((0, 1), 4)),
            False,
        ),
        # Reflexive when both sides fully partitioned and identical
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            True,
        ),
        # Inter-rank missing on one side → not aligned
        (
            NormalizedPartitioning(None, "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            False,
        ),
    ],
)
def test_is_aligned_with(spmd_engine, left, right, expected) -> None:
    """is_aligned_with checks compatible partitioning layouts."""
    br = spmd_engine.context.br()
    assert left.is_aligned_with(right, br) is expected
    assert right.is_aligned_with(left, br) is expected


def test_normalized_partitioning_eq() -> None:
    a = NormalizedPartitioning(HashScheme((0, 1), 8), "inherit")
    b = NormalizedPartitioning(HashScheme((0, 1), 8), "inherit")
    c = NormalizedPartitioning(HashScheme((0, 1), 4), "inherit")
    assert a == b
    assert a != c


def _make_select_ir(engine: pl.GPUEngine, output_columns: tuple[str, ...]):
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), engine).translate_ir()
    out_schema = {k: child.schema[k] for k in output_columns}
    exprs = tuple(
        expr.NamedExpr(name, expr.Col(child.schema[name], name))
        for name in output_columns
    )
    return Select(out_schema, exprs, should_broadcast=False, df=child)


def test_remap_partitioning_select_none_input(streaming_engine) -> None:
    assert (
        maybe_remap_partitioning(_make_select_ir(streaming_engine, ("a", "b")), None)
        is None
    )


def test_remap_partitioning_select_preserves_keys(streaming_engine) -> None:
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(
        _make_select_ir(streaming_engine, ("a", "b")), part
    )
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_groupby(streaming_engine) -> None:
    """Hash indices refer to the groupby input child; remap to groupby output columns."""
    q = (
        pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
        .group_by("a", "b")
        .agg(pl.col("c").sum())
    )
    ir = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    while isinstance(ir, (Select, Projection)):
        ir = ir.children[0]
    assert isinstance(ir, GroupBy)

    gb = ir
    key_names = tuple(ne.name for ne in gb.keys)
    child_cols = list(gb.children[0].schema.keys())
    input_indices = tuple(child_cols.index(n) for n in key_names)
    out_cols = list(gb.schema.keys())
    expected = tuple(out_cols.index(n) for n in key_names)

    part = Partitioning(inter_rank=HashScheme(input_indices, 8), local="inherit")
    result = maybe_remap_partitioning(gb, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == expected
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_hstack_appends_preserves_keys(streaming_engine) -> None:
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    d_dtype = DataType(pl.Int64())
    hstack = HStack(
        {**child.schema, "d": d_dtype},
        (expr.NamedExpr("d", expr.Literal(d_dtype, 0)),),
        should_broadcast=True,
        df=child,
    )
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(hstack, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_select_drops_key(streaming_engine) -> None:
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(_make_select_ir(streaming_engine, ("a",)), part)
    assert result is not None
    assert result.inter_rank is None
    assert result.local == "inherit"


def test_remap_partitioning_select_renamed_key(streaming_engine) -> None:
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    # Output (a_renamed, b) where a_renamed is Col("a")
    out_schema = {"a_renamed": child.schema["a"], "b": child.schema["b"]}
    exprs = (
        expr.NamedExpr("a_renamed", expr.Col(child.schema["a"], "a")),
        expr.NamedExpr("b", expr.Col(child.schema["b"], "b")),
    )
    select = Select(out_schema, exprs, should_broadcast=False, df=child)
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(select, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)  # a_renamed, b in output
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_reorder_columns(streaming_engine) -> None:
    # Select (b, a) from (a, b, c) -> partition keys (a,b) become indices (1, 0) in output
    select = _make_select_ir(streaming_engine, ("b", "a"))
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(select, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (1, 0)
    assert result.inter_rank.modulus == 8


def test_remap_partitioning_reorder_columns_projection(streaming_engine) -> None:
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    # Projection output (b, a) -> child has (a, b, c); partition keys (a,b) -> indices (1, 0)
    out_schema = {k: child.schema[k] for k in ("b", "a")}
    proj = Projection(out_schema, child)
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(proj, part, child_ir=proj.children[0])
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (1, 0)
    assert result.inter_rank.modulus == 8


def _make_ordering(context, *, key_indices=(0,), values=(100, 200), strict=False):
    stream = context.br().stream_pool.get_stream()
    df = DataFrame.from_polars(
        pl.DataFrame({f"k{i}": list(values) for i in key_indices}), stream
    )
    chunk = TableChunk.from_pylibcudf_table(
        df.table, stream, exclusive_view=False, br=context.br()
    )
    asc, before = plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    keys = [OrderKey(i, asc, before) for i in key_indices]
    return Ordering(keys, chunk, strict_boundaries=strict)


def _make_order_scheme(context, *, key_indices=(0,), values=(100, 200), strict=False):
    return OrderScheme(
        [_make_ordering(context, key_indices=key_indices, values=values, strict=strict)]
    )


@pytest.mark.parametrize(
    "keys,strict,should_match",
    [
        (
            (OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),),
            True,
            True,
        ),
        (
            (OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),),
            False,
            True,
        ),
        (
            (OrderKey(0, plc.types.Order.DESCENDING, plc.types.NullOrder.BEFORE),),
            True,
            False,
        ),
        (
            (OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.AFTER),),
            True,
            False,
        ),
        ((0,), True, True),  # plain int → matches OrderScheme by column index
    ],
)
def test_from_keys_order_scheme(spmd_engine, keys, strict, should_match):
    part = Partitioning(
        inter_rank=_make_order_scheme(spmd_engine.context, strict=strict),
        local="inherit",
    )
    result = NormalizedPartitioning.from_keys(part, nranks=4, keys=keys)
    assert isinstance(result.inter_rank_scheme, OrderScheme) == should_match


def test_is_strictly_partitioned_order_scheme(spmd_engine):
    strict = _make_order_scheme(spmd_engine.context, strict=True)
    non_strict = _make_order_scheme(spmd_engine.context, strict=False)
    assert NormalizedPartitioning(strict, "inherit").is_strictly_partitioned()
    assert not NormalizedPartitioning(non_strict, "inherit").is_strictly_partitioned()
    assert not NormalizedPartitioning(strict, non_strict).is_strictly_partitioned()


def test_is_aligned_with_order_scheme(spmd_engine):
    s1 = _make_order_scheme(spmd_engine.context, values=(100, 200), strict=True)
    s2 = _make_order_scheme(spmd_engine.context, values=(100, 200), strict=True)
    s_diff = _make_order_scheme(spmd_engine.context, values=(100, 300), strict=True)
    s_non_strict = _make_order_scheme(
        spmd_engine.context, values=(100, 200), strict=False
    )
    assert NormalizedPartitioning(s1, "inherit").is_aligned_with(
        NormalizedPartitioning(s2, "inherit"), spmd_engine.context.br()
    )
    assert not NormalizedPartitioning(s1, "inherit").is_aligned_with(
        NormalizedPartitioning(s_diff, "inherit"), spmd_engine.context.br()
    )
    assert not NormalizedPartitioning(s1, "inherit").is_aligned_with(
        NormalizedPartitioning(s_non_strict, "inherit"), spmd_engine.context.br()
    )


def test_from_keys_order_scheme_single_rank(spmd_engine):
    asc, before = plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    keys = (OrderKey(0, asc, before),)
    local_scheme = _make_order_scheme(spmd_engine.context, strict=True)
    # Single-rank: local OrderScheme promoted to inter-rank
    part = Partitioning(inter_rank=None, local=local_scheme)
    result = NormalizedPartitioning.from_keys(part, nranks=1, keys=keys)
    assert isinstance(result.inter_rank_scheme, OrderScheme)
    assert result.local_scheme == "inherit"
    # Multi-rank without inter-rank OrderScheme → no partitioning
    result_multi = NormalizedPartitioning.from_keys(part, nranks=4, keys=keys)
    assert result_multi.inter_rank_scheme is None
    # Reversed prefix: scheme has 2 keys, query has 1 → must not match
    scheme_2key = _make_order_scheme(
        spmd_engine.context, key_indices=(0, 1), strict=True
    )
    part_2key = Partitioning(inter_rank=scheme_2key, local="inherit")
    result_rev = NormalizedPartitioning.from_keys(part_2key, nranks=4, keys=keys)
    assert result_rev.inter_rank_scheme is None
    # Same check via Sequence[int] path
    result_rev_int = NormalizedPartitioning.from_keys(part_2key, nranks=4, keys=(0,))
    assert result_rev_int.inter_rank_scheme is None


def test_from_keys_order_scheme_selects_matching_ordering(spmd_engine):
    asc, before = plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    scheme = OrderScheme(
        [
            _make_ordering(spmd_engine.context, key_indices=(0,), strict=True),
            _make_ordering(spmd_engine.context, key_indices=(1,), strict=True),
        ]
    )
    part = Partitioning(inter_rank=scheme, local="inherit")

    result = NormalizedPartitioning.from_keys(
        part, nranks=4, keys=(OrderKey(1, asc, before),)
    )

    assert isinstance(result.inter_rank_scheme, OrderScheme)
    assert result.inter_rank_scheme.orderings[0].keys[0].column_index == 1
    assert [o.keys[0].column_index for o in result.inter_rank_scheme.orderings] == [
        1,
        0,
    ]


def test_from_keys_order_scheme_selects_longest_matching_ordering(spmd_engine):
    asc, before = plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    scheme = OrderScheme(
        [
            _make_ordering(spmd_engine.context, key_indices=(0,), strict=True),
            _make_ordering(spmd_engine.context, key_indices=(0, 1), strict=True),
        ]
    )
    part = Partitioning(inter_rank=scheme, local="inherit")

    result = NormalizedPartitioning.from_keys(
        part,
        nranks=4,
        keys=(OrderKey(0, asc, before), OrderKey(1, asc, before)),
    )

    assert isinstance(result.inter_rank_scheme, OrderScheme)
    assert [
        tuple(k.column_index for k in ordering.keys)
        for ordering in result.inter_rank_scheme.orderings
    ] == [(0, 1), (0,)]


def test_remap_partitioning_order_scheme_select(spmd_engine):
    part = Partitioning(
        inter_rank=_make_order_scheme(spmd_engine.context, key_indices=(0,)),
        local="inherit",
    )
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    result = maybe_remap_partitioning(_make_select_ir(engine, ("b", "a")), part)
    assert result is not None
    assert isinstance(result.inter_rank, OrderScheme)
    assert result.inter_rank.orderings[0].keys[0].column_index == 1


def test_remap_partitioning_order_scheme_updates_all_orderings(spmd_engine):
    part = Partitioning(
        inter_rank=OrderScheme(
            [
                _make_ordering(spmd_engine.context, key_indices=(0,)),
                _make_ordering(spmd_engine.context, key_indices=(1,)),
            ]
        ),
        local="inherit",
    )
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)

    result = maybe_remap_partitioning(_make_select_ir(engine, ("b", "a")), part)

    assert result is not None
    assert isinstance(result.inter_rank, OrderScheme)
    assert [o.keys[0].column_index for o in result.inter_rank.orderings] == [1, 0]


def test_remap_partitioning_order_scheme_drops_only_lost_orderings(spmd_engine):
    part = Partitioning(
        inter_rank=OrderScheme(
            [
                _make_ordering(spmd_engine.context, key_indices=(0,)),
                _make_ordering(spmd_engine.context, key_indices=(2,)),
            ]
        ),
        local="inherit",
    )
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)

    result = maybe_remap_partitioning(_make_select_ir(engine, ("b", "a")), part)

    assert result is not None
    assert isinstance(result.inter_rank, OrderScheme)
    assert [o.keys[0].column_index for o in result.inter_rank.orderings] == [1]


def test_remap_partitioning_order_scheme_drops_key(spmd_engine):
    part = Partitioning(
        inter_rank=_make_order_scheme(spmd_engine.context, key_indices=(0,)),
        local="inherit",
    )
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    result = maybe_remap_partitioning(_make_select_ir(engine, ("b",)), part)
    assert result is not None
    assert result.inter_rank is None


def test_remap_partitioning_order_scheme_adds_alias_ordering(spmd_engine):
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    child = Translator(q._ldf.visit(), engine).translate_ir()
    alias = expr.NamedExpr("a_alias", expr.Col(child.schema["a"], "a"))
    hstack = HStack(
        {**child.schema, "a_alias": child.schema["a"]},
        (alias,),
        should_broadcast=True,
        df=child,
    )
    part = Partitioning(
        inter_rank=_make_order_scheme(
            spmd_engine.context, key_indices=(0,), strict=True
        ),
        local="inherit",
    )

    result = maybe_remap_partitioning(hstack, part)

    assert result is not None
    assert isinstance(result.inter_rank, OrderScheme)
    assert [o.keys[0].column_index for o in result.inter_rank.orderings] == [0, 3]
    assert [o.strict_boundaries for o in result.inter_rank.orderings] == [True, True]


@pytest.mark.parametrize(
    "frequency,expected",
    [
        ("1ms", [1_000_000, 2_000_000]),
        ("1us", [1_234_000, 2_345_000]),
    ],
)
def test_remap_partitioning_order_scheme_adds_truncated_ordering(
    spmd_engine, frequency, expected
):
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    hstack = Translator(
        pl.LazyFrame({"DateTime": [1]})
        .with_columns(
            pl.col("DateTime")
            .cast(pl.Datetime("ns"))
            .dt.truncate(frequency)
            .cast(pl.Int64)
            .alias("ts_bucket")
        )
        ._ldf.visit(),
        engine,
    ).translate_ir()
    assert isinstance(hstack, HStack)
    part = Partitioning(
        inter_rank=_make_order_scheme(
            spmd_engine.context,
            key_indices=(0,),
            values=(1_234_567, 2_345_678),
            strict=True,
        ),
        local="inherit",
    )

    result = maybe_remap_partitioning(hstack, part, context=spmd_engine.context)

    assert result is not None
    assert isinstance(result.inter_rank, OrderScheme)
    orderings = result.inter_rank.orderings
    assert [o.keys[0].column_index for o in orderings] == [0, 1]
    assert [o.strict_boundaries for o in orderings] == [True, False]

    boundaries = orderings[1].get_boundaries(spmd_engine.context.br())
    boundary_df = DataFrame.from_table(
        boundaries.table_view(),
        ["ts_bucket"],
        [hstack.schema["ts_bucket"]],
        boundaries.stream,
    ).to_polars()
    assert boundary_df["ts_bucket"].to_list() == expected


def test_remap_partitioning_order_scheme_truncates_prefix_key(spmd_engine):
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    hstack = Translator(
        pl.LazyFrame({"venue": [1], "DateTime": [1]})
        .with_columns(
            pl.col("DateTime")
            .cast(pl.Datetime("ns"))
            .dt.truncate("1ms")
            .cast(pl.Int64)
            .alias("ts_bucket")
        )
        ._ldf.visit(),
        engine,
    ).translate_ir()
    assert isinstance(hstack, HStack)
    part = Partitioning(
        inter_rank=_make_order_scheme(
            spmd_engine.context,
            key_indices=(0, 1),
            values=(1_234_567, 2_345_678),
            strict=True,
        ),
        local="inherit",
    )

    result = maybe_remap_partitioning(hstack, part, context=spmd_engine.context)

    assert result is not None
    assert isinstance(result.inter_rank, OrderScheme)
    orderings = result.inter_rank.orderings
    assert [
        tuple(key.column_index for key in ordering.keys) for ordering in orderings
    ] == [(0, 1), (0, 2)]
    assert [o.strict_boundaries for o in orderings] == [True, False]

    boundaries = orderings[1].get_boundaries(spmd_engine.context.br())
    boundary_df = DataFrame.from_table(
        boundaries.table_view(),
        ["venue", "ts_bucket"],
        [hstack.schema["venue"], hstack.schema["ts_bucket"]],
        boundaries.stream,
    ).to_polars()
    assert boundary_df["venue"].to_list() == [1_234_567, 2_345_678]
    assert boundary_df["ts_bucket"].to_list() == [1_000_000, 2_000_000]


def test_remap_partitioning_order_scheme_ignores_unordered_truncate(spmd_engine):
    engine = pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    hstack = Translator(
        pl.LazyFrame({"venue": [1], "DateTime": [1]})
        .with_columns(
            pl.col("DateTime")
            .cast(pl.Datetime("ns"))
            .dt.truncate("1ms")
            .cast(pl.Int64)
            .alias("ts_bucket")
        )
        ._ldf.visit(),
        engine,
    ).translate_ir()
    assert isinstance(hstack, HStack)
    part = Partitioning(
        inter_rank=_make_order_scheme(
            spmd_engine.context,
            key_indices=(0,),
            strict=True,
        ),
        local="inherit",
    )

    result = maybe_remap_partitioning(hstack, part, context=spmd_engine.context)

    assert result is not None
    assert isinstance(result.inter_rank, OrderScheme)
    orderings = result.inter_rank.orderings
    assert [o.keys[0].column_index for o in orderings] == [0]
    assert [o.strict_boundaries for o in orderings] == [True]


@pytest.mark.parametrize(
    "by,descending,nulls_last",
    [
        (["x"], [False], [True]),
        (["x"], [True], [False]),
        (["x", "y"], [False, False], [True, True]),
    ],
)
def test_sort_output_metadata(spmd_engine_factory, by, descending, nulls_last) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            dynamic_planning=None,
            fallback_mode="raise",
            raise_on_fail=True,
        )
    )
    config_options = ConfigOptions.from_polars_engine(engine)
    df = pl.LazyFrame({"x": list(range(10)), "y": [i * 2 for i in range(10)]})
    q = df.sort(by=by, descending=descending, nulls_last=nulls_last)
    ir = Translator(q._ldf.visit(), engine).translate_ir()

    metadata_collector = evaluate_logical_plan(
        ir, config_options, collect_metadata=True
    )[1]
    assert metadata_collector is not None
    assert len(metadata_collector) == 1
    metadata = metadata_collector[0]

    scheme = metadata.partitioning.inter_rank
    assert isinstance(scheme, OrderScheme)
    assert metadata.partitioning.local == "inherit"

    output_cols = list(ir.schema.keys())
    ordering = scheme.orderings[0]
    assert len(ordering.keys) == len(by)
    for i, col in enumerate(by):
        assert ordering.keys[i].column_index == output_cols.index(col)
    assert ordering.strict_boundaries is True


@pytest.mark.parametrize(
    "scheme_key_count,strict,expected",
    [
        (1, True, True),  # prefix match + strict → skip
        (1, False, False),  # prefix match + non-strict → no skip
        (2, True, True),  # exact match + strict → skip
        (2, False, True),  # exact match + non-strict → skip (strict irrelevant)
    ],
)
def test_is_strictly_sorted(spmd_engine, scheme_key_count, strict, expected) -> None:
    df_lf = pl.LazyFrame({"x": list(range(5)), "y": list(range(5))})
    base_ir = Translator(df_lf._ldf.visit(), spmd_engine).translate_ir()
    asc, before = plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE

    sort_xy = Sort(
        base_ir.schema,
        (
            expr.NamedExpr("x", expr.Col(base_ir.schema["x"], "x")),
            expr.NamedExpr("y", expr.Col(base_ir.schema["y"], "y")),
        ),
        (asc, asc),
        (before, before),
        stable=False,
        zlice=None,
        df=base_ir,
    )

    ctx = spmd_engine.context
    stream = ctx.br().stream_pool.get_stream()
    keys = [OrderKey(i, asc, before) for i in range(scheme_key_count)]
    boundary_chunk = TableChunk.from_pylibcudf_table(
        DataFrame.from_polars(
            pl.DataFrame({f"k{i}": [100, 200] for i in range(scheme_key_count)}),
            stream,
        ).table,
        stream,
        exclusive_view=False,
        br=ctx.br(),
    )
    scheme = OrderScheme([Ordering(keys, boundary_chunk, strict_boundaries=strict)])
    meta = ChannelMetadata(
        3, partitioning=Partitioning(inter_rank=scheme, local="inherit")
    )

    order_keys = _sort_to_order_keys(sort_xy)
    partitioning = NormalizedPartitioning.from_keys(
        meta.partitioning, nranks=1, keys=order_keys
    )
    assert partitioning.is_strictly_sorted(order_keys) is expected
