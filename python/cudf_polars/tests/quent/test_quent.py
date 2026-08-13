# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Quent telemetry tracing."""

from __future__ import annotations

import concurrent.futures
import uuid
from typing import TYPE_CHECKING

import pytest

import polars as pl

import cudf_polars.quent
import cudf_polars.quent._logging
from cudf_polars.dsl.translate import Translator
from cudf_polars.quent._plan import build_plan, port_names_for_node
from cudf_polars.quent._types import (
    Attribute,
    Engine,
    Implementation,
    Operator,
    Plan,
    Port,
    Query,
    Worker,
    _deserialize_value,
)
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent import QuentContext
    from cudf_polars.utils.config import StreamingExecutor


def _make_worker() -> Worker:
    return Worker(
        id=uuid.uuid4(),
        engine=Engine(id=uuid.uuid4()),
        instance_name="test-worker",
    )


@pytest.mark.parametrize(
    "value,expected_variant",
    [
        (0, "U8"),
        (2**8 - 1, "U8"),
        (2**8, "U16"),
        (2**16 - 1, "U16"),
        (2**16, "U32"),
        (2**32 - 1, "U32"),
        (2**32, "U64"),
        (2**64 - 1, "U64"),
        (-(2**7), "I8"),
        (2**7 - 1, "U8"),
        (-(2**15), "I16"),
        (-(2**15) - 1, "I32"),
        (-(2**31), "I32"),
        (-(2**31) - 1, "I64"),
        (-(2**63), "I64"),
    ],
)
def test_attribute_integer_serialization_variants(
    value: int, expected_variant: str
) -> None:
    serialized = Attribute("x", value).serialize()
    assert serialized["key"] == "x"
    assert serialized["value"] == {expected_variant: value}


@pytest.mark.parametrize("value", [2**64, -(2**63) - 1])
def test_attribute_integer_serialization_overflow(value: int) -> None:
    with pytest.raises(
        ValueError,
        match="does not fit any Quent integer type",
    ):
        Attribute("x", value).serialize()


def test_attribute_serialization_uses_quent_value_envelope() -> None:
    assert Attribute("ratio", 1.5).serialize() == {
        "key": "ratio",
        "value": {"F64": 1.5},
    }
    assert Attribute("name", "scan").serialize() == {
        "key": "name",
        "value": {"String": "scan"},
    }
    assert Attribute("enabled", value=True).serialize() == {
        "key": "enabled",
        "value": {"U8": 1},
    }


def test_deserialize_value_requires_single_variant() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected Quent attribute value envelope with exactly one variant, got '2' instead.",
    ):
        _deserialize_value({"U8": 1, "I8": -1})


def test_deserialize_value_raises_on_unknown_variant() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported Quent custom attribute variant: 'UnsupportedVariant'",
    ):
        _deserialize_value({"UnsupportedVariant": "x"})


@pytest.fixture
def ir_and_config() -> tuple[IR, ConfigOptions[StreamingExecutor]]:
    q = pl.LazyFrame({"x": [1, 2]}).filter(pl.col("x") > 1)
    engine = pl.GPUEngine(executor="streaming")
    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    return ir, config_options


def test_build_plan_returns_correct_types(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    worker = _make_worker()

    plan, operators, ports, _ = build_plan(
        ir, config_options, Query(id=query_id), plan_id, worker
    )

    assert isinstance(plan, Plan)
    assert plan.id == plan_id
    assert plan.query_id == query_id
    assert plan.instance_name == "logical"
    assert plan.parent_plan_id is None
    assert plan.worker_id == worker.id
    assert len(operators) > 0
    assert len(ports) > 0
    assert all(isinstance(op, Operator) for op in operators)
    assert all(isinstance(p, Port) for p in ports)


def test_build_plan_operator_plan_ids(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    plan_id = uuid.uuid4()
    _plan, operators, _, _ = build_plan(
        ir, config_options, Query(), plan_id, _make_worker()
    )

    for op in operators:
        assert op.plan_id == plan_id


def test_build_plan_edges_reference_ports(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    plan, _operators, ports, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    port_ids = {p.id for p in ports}
    for edge in plan.edges:
        assert edge.source.id in port_ids
        assert edge.target.id in port_ids


def test_build_plan_edge_direction(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """Edges go from child 'out' port to parent input port."""
    ir, config_options = ir_and_config
    plan, _operators, _ports, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    for edge in plan.edges:
        assert edge.source.instance_name == "out"
        assert edge.target.instance_name != "out"


def test_build_plan_filter_topology(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """Filter(DataFrameScan) should produce 2 operators and 1 edge."""
    ir, config_options = ir_and_config
    plan, operators, _, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    type_names = {op.type_name for op in operators}
    assert "Filter" in type_names
    assert "DataFrameScan" in type_names
    assert len(plan.edges) == 1


def test_plan_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    plan, _, _, _ = build_plan(
        ir, config_options, Query(id=query_id), plan_id, _make_worker()
    )

    event = plan.declare(timestamp=12345)
    d = event.to_dict()
    assert d["id"] == str(plan_id)
    assert d["timestamp"] == 12345

    decl = d["data"]["Plan"]["Declaration"]
    assert decl["parent"]["query_id"] == str(query_id)
    assert decl["parent"]["plan_id"] is None
    assert decl["instance_name"] == "logical"
    assert len(decl["edges"]) == 1


def test_operator_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    _, operators, _, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    for op in operators:
        event = op.declare(timestamp=99)
        d = event.to_dict()
        assert d["id"] == str(op.id)
        decl = d["data"]["Operator"]["Declaration"]
        assert decl["plan_id"] == str(op.plan_id)
        assert decl["type_name"] == op.type_name


def test_port_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    _, _, ports, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    for port in ports:
        event = port.declare(timestamp=42)
        d = event.to_dict()
        assert d["id"] == str(port.id)
        decl = d["data"]["Port"]["Declaration"]
        assert decl["operator_id"] == str(port.operator.id)
        assert decl["instance_name"] == port.instance_name


def test_build_physical_plan(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    logical_plan_id = uuid.uuid4()
    physical_plan_id = uuid.uuid4()

    logical_plan, _, _, _ = build_plan(
        ir, config_options, Query(id=query_id), logical_plan_id, _make_worker()
    )
    assert logical_plan.instance_name == "logical"
    assert logical_plan.parent_plan_id is None

    physical_plan, _, _, _ = build_plan(
        ir,
        config_options,
        Query(id=query_id),
        physical_plan_id,
        _make_worker(),
        instance_name="physical",
        parent_plan=logical_plan,
    )
    assert physical_plan.instance_name == "physical"
    assert physical_plan.parent_plan_id == logical_plan_id
    assert physical_plan.query_id == query_id


def test_physical_plan_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    logical_plan_id = uuid.uuid4()
    physical_plan_id = uuid.uuid4()

    physical_plan, _, _, _ = build_plan(
        ir,
        config_options,
        Query(id=query_id),
        physical_plan_id,
        _make_worker(),
        instance_name="physical",
        parent_plan=Plan(
            id=logical_plan_id,
            query=Query(id=query_id),
            parent_plan=None,
            instance_name="logical",
            edges=[],
            worker=None,
        ),
    )

    event = physical_plan.declare(timestamp=99999)
    d = event.to_dict()
    decl = d["data"]["Plan"]["Declaration"]
    assert decl["instance_name"] == "physical"
    assert decl["parent"]["query_id"] == str(query_id)
    assert decl["parent"]["plan_id"] == str(logical_plan_id)


def test_build_plan_with_parent_operators(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """Physical operators reference their logical parent operators."""
    ir, config_options = ir_and_config
    _, logical_ops, _, logical_op_by_id = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    parent_operators_by_node_id = {sid: [op] for sid, op in logical_op_by_id.items()}

    _, physical_ops, _, _ = build_plan(
        ir,
        config_options,
        Query(),
        uuid.uuid4(),
        _make_worker(),
        instance_name="physical",
        parent_plan=Plan(
            id=uuid.uuid4(),
            query=Query(),
            parent_plan=None,
            instance_name="logical",
            edges=[],
            worker=None,
        ),
        parent_operators_by_node_id=parent_operators_by_node_id,
    )

    for logical_op, physical_op in zip(logical_ops, physical_ops, strict=True):
        assert len(physical_op.parent_operators) == 1
        assert physical_op.parent_operators[0] is logical_op


def test_build_plan_parent_operators_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """parent_operator_ids appear in the serialized Operator declaration."""
    ir, config_options = ir_and_config
    _, logical_ops, _, logical_op_by_id = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    parent_operators_by_node_id = {sid: [op] for sid, op in logical_op_by_id.items()}

    _, physical_ops, _, _ = build_plan(
        ir,
        config_options,
        Query(),
        uuid.uuid4(),
        _make_worker(),
        instance_name="physical",
        parent_operators_by_node_id=parent_operators_by_node_id,
    )

    for logical_op, physical_op in zip(logical_ops, physical_ops, strict=True):
        d = physical_op.declare(timestamp=1).to_dict()
        parent_ids = d["data"]["Operator"]["Declaration"]["parent_operator_ids"]
        assert parent_ids == [str(logical_op.id)]


def test_build_plan_without_parent_operators_has_empty_list(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    _, operators, _, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    for op in operators:
        assert op.parent_operators == []


def test_port_names_for_node_leaf() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1", children=[], schema={}, properties={}, type="Scan"
    )
    assert port_names_for_node(len(node.children), node.type) == ("out",)


def test_port_names_for_node_single_child() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1", children=["2"], schema={}, properties={}, type="Filter"
    )
    assert port_names_for_node(len(node.children), node.type) == ("out", "in")


def test_port_names_for_node_join() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1", children=["2", "3"], schema={}, properties={}, type="Join"
    )
    assert port_names_for_node(len(node.children), node.type) == (
        "out",
        "left",
        "right",
    )


def test_port_names_for_node_multi_child() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1",
        children=["2", "3", "4"],
        schema={},
        properties={},
        type="Union",
    )

    assert port_names_for_node(len(node.children), node.type) == (
        "out",
        "in_0",
        "in_1",
        "in_2",
    )


def test_lower_ir_graph_with_node_map() -> None:
    from cudf_polars.streaming.parallel import lower_ir_graph_with_node_map
    from cudf_polars.streaming.statistics import collect_statistics

    q = pl.LazyFrame({"x": [1, 2]}).filter(pl.col("x") > 1)
    engine = pl.GPUEngine(executor="streaming")
    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_statistics(
        ir, config_options, concurrent.futures.ThreadPoolExecutor()
    )

    lowering, node_map = lower_ir_graph_with_node_map(ir, config_options, stats)

    assert lowering.optimized is ir
    assert len(node_map) > 0
    for physical_sid, logical_sids in node_map.items():
        assert isinstance(physical_sid, str)
        assert isinstance(logical_sids, list)
        assert all(isinstance(s, str) for s in logical_sids)


def test_engine_lifecycle() -> None:
    engine_id = uuid.uuid4()
    impl = Implementation()
    engine = Engine(id=engine_id, implementation=impl)

    init_event = engine._init()
    d = init_event.to_dict()
    init = d["data"]["Engine"]["Init"]
    assert init["implementation"]["name"] == "cudf-polars"
    assert init["instance_name"].startswith("cudf-polars-")

    exit_event = engine._exit()
    d = exit_event.to_dict()
    assert d["data"]["Engine"]["Exit"] is None


def test_worker_lifecycle() -> None:
    engine_id = uuid.uuid4()
    worker_id = uuid.uuid4()
    worker = Worker(id=worker_id, engine=Engine(id=engine_id), instance_name="rank-0")

    init_event = worker._init()
    d = init_event.to_dict()
    assert d["id"] == str(worker_id)
    init_data = d["data"]["Worker"]["Init"]
    assert init_data["parent_engine_id"] == str(engine_id)
    assert init_data["instance_name"] == "rank-0"

    exit_event = worker._exit()
    d = exit_event.to_dict()
    assert d["id"] == str(worker_id)
    assert d["data"]["Worker"]["Exit"] is None


def test_query_lifecycle() -> None:
    query_id = uuid.uuid4()
    group_id = uuid.uuid4()
    query = Query(id=query_id)

    init_event = query._init(query_group=cudf_polars.quent.QueryGroup(id=group_id))
    assert init_event.to_dict()["data"]["Query"]["seq"] == 0
    assert query._planning().to_dict()["data"]["Query"]["seq"] == 1
    assert query._executing().to_dict()["data"]["Query"]["seq"] == 2
    assert query._exit().to_dict()["data"]["Query"]["seq"] == 3


@pytest.fixture
def quent_context() -> QuentContext:
    return cudf_polars.quent.QuentContext(
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )


def test_quent_context_serialization() -> None:
    quent_context = cudf_polars.quent.QuentContext(
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )
    data = quent_context.serialize()

    new = cudf_polars.quent.QuentContext.deserialize(data)
    assert new == quent_context


def test_quent_context_serialization_with_custom_attributes() -> None:
    engine = Engine(
        implementation=Implementation(
            name="test-impl",
            version="1.2.3",
            custom_attributes=[
                Attribute("count", 3),
                Attribute("ratio", 1.5),
                Attribute("name", "demo"),
                Attribute("optional", None),
            ],
        )
    )
    quent_context = cudf_polars.quent.QuentContext(
        engine=engine,
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )

    data = quent_context.serialize()
    new = cudf_polars.quent.QuentContext.deserialize(data)

    assert new == quent_context


def test_emit_query_group_events_idempotent(quent_context: QuentContext):
    pytest.importorskip("structlog")
    logger = cudf_polars.quent._logging.QuentLogger()
    quent_context._emit_query_group_events(logger)
    quent_context._emit_query_group_events(logger)
    assert len(logger._buffer) == 1


def test_serialize_list_raises():
    with pytest.raises(NotImplementedError, match="not supported yet"):
        Attribute("list", [1, 2]).serialize()


def test_serialize_dict_raises():
    with pytest.raises(NotImplementedError, match="not supported yet"):
        Attribute("dict", {"a": 1, "b": 2}).serialize()


def test_quent_serialize_none():
    assert Attribute("none", None).serialize() == {
        "key": "none",
        "value": None,
    }
