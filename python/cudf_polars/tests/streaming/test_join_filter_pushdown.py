# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Cache, DataFrameScan, Distinct, Join, Select, Slice
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.column_domain import ColumnRef
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.join_filter_pushdown import (
    CompositeCandidate,
    Decision,
    PlanFacts,
    SimpleCandidate,
    _select_candidate,
    _smallest_node_containing_all,
    analyze_plan,
    apply_candidate,
    contains_node,
    optimize_join_filter_pushdown,
    semijoin_pushdown_candidates,
)
from cudf_polars.streaming.parallel import optimize_with_stats, remove_cache_nodes
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    import concurrent.futures
    from typing import Any

    from cudf_polars.dsl.ir import IR
    from cudf_polars.engine.spmd import SPMDEngine


@pytest.fixture
def engine(spmd_engine_factory) -> SPMDEngine:
    """Return an SPMD engine configured for join-domain prefilter tests."""
    return spmd_engine_factory(
        StreamingOptions(
            join_filter_pushdown={"threshold": 0.5},
            raise_on_fail=True,
        )
    )


def make_config(
    *, dynamic_planning: bool = True, join_filter_pushdown: bool = True
) -> ConfigOptions:
    executor_options: dict[str, Any] = {
        "join_filter_pushdown": {"trace": False} if join_filter_pushdown else None
    }
    if not dynamic_planning:
        executor_options["dynamic_planning"] = None
    return ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options=executor_options,
        )
    )


def find_joins(ir: IR, how: str | None = None) -> list[Join]:
    return [
        node
        for node in traversal([ir])
        if isinstance(node, Join) and (how is None or node.options[0] == how)
    ]


def translate_query(query: pl.LazyFrame, engine: SPMDEngine) -> IR:
    """Translate a public Polars query and remove logical Cache nodes."""
    t = Translator(query._ldf.visit(), engine)
    root = t.translate_ir()
    assert not t.errors
    return remove_cache_nodes(root)


def dataframe_scan(ir: IR, column: str) -> DataFrameScan:
    """Return the unique in-memory scan containing ``column``."""
    (match,) = (
        node
        for node in traversal([ir])
        if isinstance(node, DataFrameScan) and column in node.schema
    )
    return match


def join_key_names(join: Join) -> tuple[str, ...]:
    """Return the column names used on the left of a simple-column join."""
    names = tuple(key.value.name for key in join.left_on if isinstance(key.value, Col))
    assert len(names) == len(join.left_on)
    return names


@pytest.fixture
def simple_query() -> pl.LazyFrame:
    """Return a query with a small selective join domain."""
    part = (
        pl.LazyFrame(
            {
                "p_partkey": [1, 99],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("p_partkey")
    )
    lineitem = pl.LazyFrame(
        {
            "l_partkey": [i % 10 for i in range(20)],
            "l_suppkey": range(20),
        }
    )
    return part.join(lineitem, left_on="p_partkey", right_on="l_partkey")


def test_simple_prefilter_filters_large_side(
    simple_query: pl.LazyFrame, engine: SPMDEngine
) -> None:
    root = translate_query(simple_query, engine)

    assert isinstance(root, Join)
    part_ir, _ = root.children
    lineitem_ir = dataframe_scan(root, "l_partkey")
    facts = analyze_plan(root, StatsCollector())
    decision = _select_candidate(root, 0.5, facts)

    assert decision.reason == "applied"
    assert isinstance(decision.candidate, SimpleCandidate)
    optimized = apply_candidate(root, decision.candidate)

    assert isinstance(optimized, Join)
    assert optimized.options[0] == "Inner"
    semis = find_joins(optimized, "Semi")
    assert len(semis) == 1
    assert semis[0].children[0] is lineitem_ir
    assert not find_joins(part_ir, "Semi")
    assert_gpu_result_equal(simple_query, engine=engine, check_row_order=False)


def test_filter_pushdown_is_independent_of_dynamic_planning(
    simple_query: pl.LazyFrame,
    engine: SPMDEngine,
) -> None:
    root = translate_query(simple_query, engine)

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        make_config(dynamic_planning=False),
    )

    assert find_joins(optimized, "Semi")


def test_filter_pushdown_can_be_disabled(
    simple_query: pl.LazyFrame, engine: SPMDEngine
) -> None:
    root = translate_query(simple_query, engine)

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        make_config(join_filter_pushdown=False),
    )

    assert optimized is root


@pytest.mark.parametrize(
    "nulls_equal", [False, True], ids=["nulls_not_equal", "nulls_equal"]
)
def test_nullable_join_keys_preserve_results(
    nulls_equal: bool,  # noqa: FBT001
    engine: SPMDEngine,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    domain = pl.LazyFrame(
        {
            "key": [None, 1, 2, 9],
            "active": [True, True, True, False],
        }
    ).filter("active")
    target = pl.LazyFrame(
        {
            "key": [None, 1, 2, 3] * 10,
            "value": range(40),
        }
    )
    query = domain.join(target, on="key", nulls_equal=nulls_equal)

    ir = Translator(query._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    optimized = optimize_join_filter_pushdown(
        ir,
        collect_statistics(ir, config, parquet_stats_executor),
        config,
    )

    semi_joins = find_joins(optimized, "Semi")
    assert semi_joins
    assert all(join.options[1] is nulls_equal for join in semi_joins)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_prefilter_does_not_move_below_distinct_on_non_subset_column(
    engine: SPMDEngine,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    target = pl.LazyFrame(
        {
            "group": [1, 1] * 100,
            "key": [0, 1] * 100,
        }
    ).unique(subset="group", keep="first", maintain_order=True)
    domain = pl.LazyFrame(
        {
            "key": [1, 2],
            "active": [True, False],
        }
    ).filter("active")
    query = target.join(domain, on="key")

    ir = Translator(query._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    optimized = optimize_join_filter_pushdown(
        ir,
        collect_statistics(ir, config, parquet_stats_executor),
        config,
    )

    semis = find_joins(optimized, "Semi")
    assert any(isinstance(semi.children[0], Distinct) for semi in semis)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_no_simple_filter_pushdown_when_domain_is_not_selective(
    engine: SPMDEngine,
) -> None:
    supplier = pl.LazyFrame({"s_suppkey": range(3)})
    lineitem = pl.LazyFrame({"l_suppkey": [i % 3 for i in range(20)]})
    query = supplier.join(
        lineitem,
        left_on="s_suppkey",
        right_on="l_suppkey",
    )
    root = translate_query(query, engine)
    stats = StatsCollector()
    assert isinstance(root, Join)
    decision = _select_candidate(root, 0.5, analyze_plan(root, stats))

    optimized = optimize_join_filter_pushdown(
        root,
        stats,
        ConfigOptions.from_polars_engine(engine),
    )

    assert decision == Decision(reason="no_profitable_domain")
    assert optimized is root
    assert not find_joins(optimized, "Semi")
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_composite_filter_pushdown_constrains_domain_first(
    engine: SPMDEngine,
) -> None:
    nation = (
        pl.LazyFrame(
            {
                "n_nationkey": range(10),
                "active": [True] * 5 + [False] * 5,
            }
        )
        .filter("active")
        .select("n_nationkey")
    )
    orders = pl.LazyFrame(
        {
            "o_orderkey": range(90),
            "n_nationkey": [i % 10 for i in range(90)],
        }
    )
    lineitem = pl.LazyFrame(
        {
            "l_orderkey": [i % 90 for i in range(180)],
            "l_suppkey": [i % 30 for i in range(180)],
        }
    )
    supplier = pl.LazyFrame(
        {
            "s_suppkey": range(30),
            "s_nationkey": [i % 10 for i in range(30)],
        }
    )
    query = (
        nation.join(orders, on="n_nationkey")
        .join(
            lineitem,
            left_on="o_orderkey",
            right_on="l_orderkey",
            maintain_order="left",
        )
        .join(
            supplier,
            left_on=("l_suppkey", "n_nationkey"),
            right_on=("s_suppkey", "s_nationkey"),
        )
    )
    root = remove_cache_nodes(Translator(query._ldf.visit(), engine).translate_ir())
    config = ConfigOptions.from_polars_engine(engine)
    stats = StatsCollector()

    assert isinstance(root, Join)
    order_lineitem, supplier_ir = root.children
    assert isinstance(order_lineitem, Join)
    lineitem_ir = order_lineitem.children[1]
    decision = _select_candidate(root, 0.5, analyze_plan(root, stats))
    optimized = optimize_join_filter_pushdown(root, stats, config)

    assert decision.reason == "applied"
    assert isinstance(decision.candidate, CompositeCandidate)
    semis = find_joins(optimized, "Semi")
    assert isinstance(optimized, Join)
    assert optimized.options[0] == "Inner"
    assert optimized.children[1] is supplier_ir
    assert any(semi.children[0] is supplier_ir for semi in semis)
    assert any(semi.children[0] is lineitem_ir for semi in semis)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_prefilter_uses_cheaper_source_domain_and_skips_expensive_domain(
    engine: SPMDEngine,
) -> None:
    part = (
        pl.LazyFrame(
            {
                "p_partkey": range(60),
                "part_active": [True] * 30 + [False] * 30,
            }
        )
        .filter("part_active")
        .select("p_partkey")
    )
    partsupp = pl.LazyFrame(
        {
            "ps_partkey": [i % 60 for i in range(120)],
            "ps_suppkey": [i % 30 for i in range(120)],
        }
    )
    supplier = pl.LazyFrame({"s_suppkey": range(30)})
    lineitem = pl.LazyFrame(
        {
            "l_partkey": [i % 60 for i in range(1_800)],
            "l_suppkey": [i % 30 for i in range(1_800)],
            "l_orderkey": [i % 900 for i in range(1_800)],
        }
    )
    orders = pl.LazyFrame({"o_orderkey": range(900)})
    query = (
        part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(
            lineitem,
            left_on=("p_partkey", "ps_suppkey"),
            right_on=("l_partkey", "l_suppkey"),
        )
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
    )
    root = translate_query(query, engine)

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    part_ir = dataframe_scan(root, "p_partkey")
    supplier_ir = dataframe_scan(root, "s_suppkey")
    lineitem_ir = dataframe_scan(root, "l_orderkey")
    orders_ir = dataframe_scan(root, "o_orderkey")
    semis = find_joins(optimized, "Semi")
    partkey_semis = [
        semi
        for semi in semis
        if semi.children[0] is lineitem_ir and join_key_names(semi) == ("l_partkey",)
    ]
    assert partkey_semis
    assert not any(semi.children[0] is orders_ir for semi in semis)
    assert contains_node(partkey_semis[0].children[1], part_ir)
    assert not contains_node(partkey_semis[0].children[1], supplier_ir)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_source_only_domain_does_not_stack_on_prefiltered_source(
    engine: SPMDEngine,
) -> None:
    part = (
        pl.LazyFrame(
            {
                "p_partkey": range(60),
                "part_active": [True] * 30 + [False] * 30,
            }
        )
        .filter("part_active")
        .select("p_partkey")
    )
    lineitem = pl.LazyFrame(
        {
            "l_partkey": [i % 60 for i in range(1_800)],
            "l_orderkey": [i % 150 for i in range(1_800)],
        }
    )
    orders = (
        pl.LazyFrame(
            {
                "o_orderkey": range(150),
                "order_active": [True] * 75 + [False] * 75,
            }
        )
        .filter("order_active")
        .select("o_orderkey")
    )
    query = part.join(lineitem, left_on="p_partkey", right_on="l_partkey").join(
        orders, left_on="l_orderkey", right_on="o_orderkey"
    )
    root = translate_query(query, engine)

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    lineitem_ir = dataframe_scan(root, "l_partkey")
    lineitem_semis = [
        semi
        for semi in find_joins(optimized, "Semi")
        if semi.children[0] is lineitem_ir
    ]
    assert any(join_key_names(semi) == ("l_partkey",) for semi in lineitem_semis)
    assert not any(join_key_names(semi) == ("l_orderkey",) for semi in lineitem_semis)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_derived_selectivity_propagates_through_rewritten_children(
    engine: SPMDEngine,
) -> None:
    region = (
        pl.LazyFrame(
            {
                "r_regionkey": [0, 1],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("r_regionkey")
    )
    nation = pl.LazyFrame(
        {
            "n_nationkey": range(10),
            "n_regionkey": [i % 2 for i in range(10)],
        }
    )
    customer = pl.LazyFrame(
        {
            "c_custkey": range(40),
            "c_nationkey": [i % 10 for i in range(40)],
        }
    )
    orders = pl.LazyFrame(
        {
            "o_orderkey": range(200),
            "o_custkey": [i % 40 for i in range(200)],
        }
    )
    query = (
        region.join(nation, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
    )
    root = translate_query(query, engine)

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    semis = find_joins(optimized, "Semi")
    expected_targets = {
        dataframe_scan(root, "n_nationkey"),
        dataframe_scan(root, "c_custkey"),
        dataframe_scan(root, "o_orderkey"),
    }
    assert expected_targets <= {semi.children[0] for semi in semis}
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_rewritten_domain_filters_other_side_instead_of_stacking(
    engine: SPMDEngine,
) -> None:
    part = (
        pl.LazyFrame(
            {
                "p_partkey": range(6),
                "part_active": [True] * 3 + [False] * 3,
            }
        )
        .filter("part_active")
        .select("p_partkey")
    )
    lineitem = pl.LazyFrame(
        {
            "l_orderkey": [i % 15 for i in range(180)],
            "l_partkey": [i % 6 for i in range(180)],
            "l_suppkey": [i % 3 for i in range(180)],
        }
    )
    supplier = pl.LazyFrame({"s_suppkey": range(3)})
    orders = (
        pl.LazyFrame(
            {
                "o_orderkey": range(15),
                "order_active": [True] * 8 + [False] * 7,
            }
        )
        .filter("order_active")
        .select("o_orderkey")
    )
    query = (
        part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
    )
    root = translate_query(query, engine)

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    lineitem_ir = dataframe_scan(root, "l_orderkey")
    orders_ir = dataframe_scan(root, "o_orderkey")
    semis = find_joins(optimized, "Semi")
    assert sum(semi.children[0] is lineitem_ir for semi in semis) == 1
    assert not any(semi.children[0] is orders_ir for semi in semis)
    assert not any(
        isinstance(semi.children[0], Join) and semi.children[0].options[0] == "Semi"
        for semi in semis
    )
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_target_source_follows_join_key_through_rename(
    engine: SPMDEngine,
) -> None:
    big = pl.LazyFrame(
        {
            "left_key": range(20),
            "other": [i % 5 for i in range(20)],
        }
    )
    renamed_big = big.select(pl.col("left_key").alias("foo"), "other")
    small = pl.LazyFrame(
        {
            "left_key": range(10),
            "other2": [i % 5 for i in range(10)],
        }
    )
    domain = (
        pl.LazyFrame(
            {
                "domain_key": [1, 99],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("domain_key")
    )
    query = renamed_big.join(
        small,
        left_on="other",
        right_on="other2",
        maintain_order="left",
    ).join(
        domain,
        left_on="left_key",
        right_on="domain_key",
    )
    root = remove_cache_nodes(Translator(query._ldf.visit(), engine).translate_ir())

    assert isinstance(root, Join)
    joined = root.children[0]
    assert isinstance(joined, Join)
    renamed_big_ir, small_ir = joined.children
    assert isinstance(renamed_big_ir, Select)
    big_ir = renamed_big_ir.children[0]
    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    semis = find_joins(optimized, "Semi")
    assert any(semi.children[0] is small_ir for semi in semis)
    assert not any(semi.children[0] is big_ir for semi in semis)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_domain_source_follows_join_key_through_rename(
    engine: SPMDEngine,
) -> None:
    target = pl.LazyFrame({"target_key": range(20)})
    unrelated = pl.LazyFrame(
        {
            "domain_key": [100, 101],
            "other": [0, 1],
            "active": [True, False],
        }
    )
    renamed_unrelated = unrelated.filter("active").select(
        pl.col("domain_key").alias("foo"), "other"
    )
    domain_source = pl.LazyFrame(
        {
            "domain_key": range(1, 6),
            "other2": range(5),
        }
    )
    domain = renamed_unrelated.join(
        domain_source,
        left_on="other",
        right_on="other2",
        maintain_order="left",
    )
    query = target.join(
        domain,
        left_on="target_key",
        right_on="domain_key",
    )
    root = remove_cache_nodes(Translator(query._ldf.visit(), engine).translate_ir())

    assert isinstance(root, Join)
    target_ir, domain_ir = root.children
    assert isinstance(domain_ir, Join)
    renamed_unrelated_ir, domain_source_ir = domain_ir.children
    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    semi = next(
        semi for semi in find_joins(optimized, "Semi") if semi.children[0] is target_ir
    )
    selected_domain = semi.children[1]
    assert isinstance(selected_domain, Select)
    rewritten_domain_source = selected_domain.children[0]
    assert isinstance(rewritten_domain_source, Join)
    assert rewritten_domain_source.options[0] == "Semi"
    assert rewritten_domain_source.children[0] is domain_source_ir
    assert rewritten_domain_source.children[0] is not renamed_unrelated_ir
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_composite_domain_columns_follow_renames(engine: SPMDEngine) -> None:
    query = pl.LazyFrame(
        {
            "raw_key": range(10),
            "raw_constraint": range(10),
        }
    ).select(
        pl.col("raw_key").alias("domain_key"),
        pl.col("raw_constraint").alias("domain_constraint"),
    )
    renamed = translate_query(query, engine)
    source = dataframe_scan(renamed, "raw_key")

    analyzed = analyze_plan(renamed, StatsCollector())
    facts = PlanFacts(
        row_estimates={renamed: 20, source: 10},
        source_facts=analyzed.source_facts,
        selective_nodes=analyzed.selective_nodes,
        column_lineages=analyzed.column_lineages,
        refcounts=analyzed.refcounts,
    )
    producer = _smallest_node_containing_all(
        renamed, ("domain_key", "domain_constraint"), facts
    )

    assert producer is not None
    assert producer.node is source
    assert producer.columns == ("raw_key", "raw_constraint")


def test_composite_domain_columns_do_not_reconverge_after_join(
    engine: SPMDEngine,
) -> None:
    source = pl.LazyFrame({"key": [1, 1, 2], "value": [10, 20, 30]})
    query = source.join(source, on="key", suffix="_right")
    joined = Translator(query._ldf.visit(), engine).translate_ir()

    assert isinstance(joined, Join)
    assert isinstance(joined.children[0], Cache)
    assert joined.children[0] is joined.children[1]
    joined = remove_cache_nodes(joined)
    assert isinstance(joined, Join)
    assert joined.children[0] is joined.children[1]

    facts = analyze_plan(joined, StatsCollector())
    producer = _smallest_node_containing_all(joined, ("value", "value_right"), facts)

    candidates = tuple(semijoin_pushdown_candidates(facts, joined, "value"))
    assert candidates[0] == (ColumnRef(joined, "value"), ())
    assert len(candidates) >= 2
    assert all(path == (0,) * len(path) for _, path in candidates[1:])
    assert producer is not None
    assert producer.node is joined
    assert producer.columns == ("value", "value_right")
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_contains_node_uses_dag_equality(engine: SPMDEngine) -> None:
    query = pl.LazyFrame({"key": range(3)}).filter(pl.col("key") >= 0).slice(0, 2)
    root = translate_query(query, engine)

    assert isinstance(root, Slice)
    source = root.children[0]
    equal_source = source.reconstruct(source.children)

    assert source is not equal_source
    assert source == equal_source
    assert contains_node(root, equal_source)


def test_plan_facts_share_lineage_suffixes_across_shared_dag(
    engine: SPMDEngine,
) -> None:
    source = pl.LazyFrame({"raw_key": range(10)})
    query = source.select(pl.col("raw_key").alias("left_key")).join(
        source.select(pl.col("raw_key").alias("right_key")),
        left_on="left_key",
        right_on="right_key",
    )
    root = translate_query(query, engine)

    assert isinstance(root, Join)
    left, right = root.children
    source_ir = dataframe_scan(root, "raw_key")
    facts = analyze_plan(root, StatsCollector())
    left_lineage = facts.column_lineages[ColumnRef(left, "left_key")]
    right_lineage = facts.column_lineages[ColumnRef(right, "right_key")]
    source_lineage = facts.column_lineages[ColumnRef(source_ir, "raw_key")]

    assert left_lineage.column == ColumnRef(left, "left_key")
    assert right_lineage.column == ColumnRef(right, "right_key")
    while left_lineage.source is not source_lineage:
        assert left_lineage.source is not None
        left_lineage = left_lineage.source
    while right_lineage.source is not source_lineage:
        assert right_lineage.source is not None
        right_lineage = right_lineage.source
    assert left_lineage.source is right_lineage.source
    assert source_lineage.source is None


def test_target_prefilter_does_not_move_below_slice(engine: SPMDEngine) -> None:
    target = (
        pl.LazyFrame({"target_key": range(20)})
        .filter(pl.col("target_key") >= 0)
        .slice(0, 10)
    )
    domain = (
        pl.LazyFrame(
            {
                "domain_key": [1, 99],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("domain_key")
    )
    query = target.join(domain, left_on="target_key", right_on="domain_key")
    root = translate_query(query, engine)

    assert isinstance(root, Join)
    sliced = root.children[0]
    assert isinstance(sliced, Slice)
    target_ir = dataframe_scan(root, "target_key")
    stats = StatsCollector()
    facts = analyze_plan(root, stats)
    lineage = facts.column_lineages[ColumnRef(sliced, "target_key")]
    assert lineage.column == ColumnRef(sliced, "target_key")
    assert tuple(semijoin_pushdown_candidates(facts, sliced, "target_key")) == (
        (ColumnRef(sliced, "target_key"), ()),
    )

    optimized = optimize_join_filter_pushdown(
        root,
        stats,
        ConfigOptions.from_polars_engine(engine),
    )

    semis = find_joins(optimized, "Semi")
    assert any(semi.children[0] is sliced for semi in semis)
    assert not any(semi.children[0] is target_ir for semi in semis)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_target_replacement_does_not_rewrite_shared_domain_side(
    engine: SPMDEngine,
) -> None:
    shared = pl.LazyFrame(
        {
            "target_key": range(20),
            "other": [i % 2 for i in range(20)],
        }
    )
    domain_source = (
        pl.LazyFrame(
            {
                "domain_key": [1, 99],
                "other2": [0, 1],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("domain_key", "other2")
    )
    domain = shared.join(
        domain_source,
        left_on=pl.col("other").cast(pl.Int32),
        right_on=pl.col("other2").cast(pl.Int32),
    )
    query = shared.join(domain, left_on="target_key", right_on="domain_key")
    root = translate_query(query, engine)

    assert isinstance(root, Join)
    shared_ir, domain_ir = root.children
    assert isinstance(domain_ir, Join)
    assert domain_ir.children[0] is shared_ir

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    assert isinstance(optimized, Join)
    filtered, unfiltered_domain = optimized.children
    assert unfiltered_domain is domain_ir
    assert domain_ir.children[0] is shared_ir
    semis = find_joins(filtered, "Semi")
    assert len(semis) == 1
    assert semis[0].children[0] is shared_ir
    assert not find_joins(unfiltered_domain, "Semi")
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_target_prefilter_rewrites_only_selected_self_join_edge(
    engine: SPMDEngine,
) -> None:
    source = pl.LazyFrame(
        {
            "key": [1, 1, 2, 2],
            "value": [10, 20, 30, 40],
        }
    )
    domain = (
        pl.LazyFrame(
            {
                "domain_value": [10, 999],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("domain_value")
    )
    query = source.join(source, on="key", suffix="_right").join(
        domain,
        left_on="value",
        right_on="domain_value",
    )
    translated = Translator(query._ldf.visit(), engine).translate_ir()

    assert isinstance(translated, Join)
    translated_self_join = translated.children[0]
    assert isinstance(translated_self_join, Join)
    shared_cache = translated_self_join.children[0]
    assert isinstance(shared_cache, Cache)
    assert translated_self_join.children[1] is shared_cache
    source_ir = shared_cache.children[0]

    optimized = optimize_with_stats(
        translated,
        ConfigOptions.from_polars_engine(engine),
        StatsCollector(),
    )

    assert isinstance(optimized, Join)
    rewritten_self_join = optimized.children[0]
    assert isinstance(rewritten_self_join, Join)
    filtered, unfiltered = rewritten_self_join.children
    assert unfiltered is source_ir
    filtered_semis = find_joins(filtered, "Semi")
    assert len(filtered_semis) == 1
    assert not find_joins(unfiltered, "Semi")
    # The shared node is a valid insertion point, but its children are not:
    # Only this consumer should be wrapped by the semi-join.
    assert filtered_semis[0].children[0] is source_ir
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_internal_prefilter_rewrites_shared_subplan_once(
    engine: SPMDEngine,
) -> None:
    domain = (
        pl.LazyFrame(
            {
                "domain_key": [1, 99],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("domain_key")
    )
    target = pl.LazyFrame(
        {
            "target_key": [i % 10 for i in range(20)],
            "value": range(20),
        }
    )
    shared = domain.join(
        target,
        left_on="domain_key",
        right_on="target_key",
    )
    query = shared.join(shared, on="domain_key", suffix="_right")
    translated = Translator(query._ldf.visit(), engine).translate_ir()

    assert isinstance(translated, Join)
    shared_cache = translated.children[0]
    assert isinstance(shared_cache, Cache)
    assert translated.children[1] is shared_cache
    original_shared = shared_cache.children[0]
    assert len(find_joins(original_shared, "Inner")) == 1
    target_ir = dataframe_scan(original_shared, "target_key")

    optimized = optimize_with_stats(
        translated,
        ConfigOptions.from_polars_engine(engine),
        StatsCollector(),
    )

    assert isinstance(optimized, Join)
    rewritten_left, rewritten_right = optimized.children
    assert rewritten_left is rewritten_right
    assert rewritten_left is not original_shared
    (internal_semi,) = find_joins(rewritten_left, "Semi")
    assert internal_semi.children[0] is target_ir
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


@pytest.mark.parametrize("how", ["left", "right", "cross", "full"])
def test_no_filter_pushdown_for_unsupported_joins(
    how: Any,
    engine: SPMDEngine,
) -> None:
    part = (
        pl.LazyFrame(
            {
                "p_partkey": [1, 99],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("p_partkey")
    )
    lineitem = pl.LazyFrame({"l_partkey": [i % 10 for i in range(20)]})
    if how == "cross":
        query = part.join(lineitem, how=how)
    else:
        query = part.join(
            lineitem,
            left_on="p_partkey",
            right_on="l_partkey",
            how=how,
        )
    root = translate_query(query, engine)

    optimized = optimize_join_filter_pushdown(
        root,
        StatsCollector(),
        ConfigOptions.from_polars_engine(engine),
    )

    assert optimized is root
    assert not find_joins(optimized, "Semi")
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)
