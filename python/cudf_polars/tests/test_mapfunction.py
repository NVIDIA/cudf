# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc

import cudf_polars.streaming.parallel  # noqa: F401
from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import DataFrameScan, IRExecutionContext, MapFunction
from cudf_polars.dsl.translate import Translator
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_140

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.dispatch import State


def test_explode_multiple_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]})
    q = df.explode("a", "b")

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("column", ["a", "b"])
def test_explode_single(engine: pl.GPUEngine, column):
    df = pl.LazyFrame(
        {
            "a": [[1, 2], [3, 4], None],
            "b": [[5, 6], [7, 8], [9, 10]],
            "c": [None, 11, 12],
        }
    )
    q = df.explode(column)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("mapping", [{"b": "a"}, {"a": "c", "b": "c"}])
def test_rename_duplicate_raises(engine: pl.GPUEngine, mapping):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

    q = df.rename(mapping)

    with pytest.raises(pl.exceptions.DuplicateError, match="is duplicate"):
        assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "mapping", [{}, {"b": "c"}, {"b": "a", "a": "b"}, {"a": "c", "b": "d"}]
)
def test_rename_columns(engine: pl.GPUEngine, mapping):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

    q = df.rename(mapping)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("index", [None, ["a"], ["d", "a"]])
@pytest.mark.parametrize("variable_name", [None, "names"])
@pytest.mark.parametrize("value_name", [None, "unpivoted"])
def test_unpivot(engine: pl.GPUEngine, index, variable_name, value_name):
    df = pl.LazyFrame(
        {
            "a": ["x", "y", "z"],
            "b": pl.Series([1, 3, 5], dtype=pl.Int16),
            "c": pl.Series([2, 4, 6], dtype=pl.Float32),
            "d": ["a", "b", "c"],
        }
    )
    q = df.unpivot(
        ["c", "b"], index=index, variable_name=variable_name, value_name=value_name
    )

    assert_gpu_result_equal(q, engine=engine)


def test_unpivot_defaults(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "a": pl.Series([11, 12, 13], dtype=pl.UInt16),
            "b": pl.Series([1, 3, 5], dtype=pl.Int16),
            "c": pl.Series([2, 4, 6], dtype=pl.Float32),
            "d": ["a", "b", "c"],
        }
    )
    q = df.unpivot(index="d")
    assert_gpu_result_equal(q, engine=engine)


def test_with_row_index_defaults(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "a": [1, 3, 5],
            "b": [2, 4, 6],
        }
    )
    q = lf.with_row_index()
    assert_gpu_result_equal(q, engine=engine)


def test_unique_hash():
    # https://github.com/NVIDIA/cudf/pull/19121#issuecomment-2959305678
    a = pl.LazyFrame({"a": [1, 2, 3]}).rename({"a": "A"})
    b = pl.LazyFrame({"a": [4, 5, 6]}).rename({"a": "A"})
    ir_a = Translator(a._ldf.visit(), pl.GPUEngine()).translate_ir()
    ir_b = Translator(b._ldf.visit(), pl.GPUEngine()).translate_ir()

    assert hash(ir_a) != hash(ir_b)


def test_set_sorted_then_inner_join(
    engine: pl.GPUEngine, request: pytest.FixtureRequest
):
    if POLARS_VERSION_LT_140:
        request.applymarker(
            pytest.mark.xfail(reason="set_sorted lowers to unsupported hint ir")
        )
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})

    q = df.set_sorted("a").join(
        pl.LazyFrame({"a": [2, 4], "b": [20, 40]}), on="a", how="inner"
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_hint_sorted_marks_column_metadata(descending, nulls_last) -> None:
    schema = {
        "a": DataType(pl.Int64()),
        "b": DataType(pl.Int64()),
    }
    child = DataFrameScan(
        schema,
        pl.DataFrame(
            {
                "a": [2, None, 1],
                "b": [3, 1, 2],
            }
        )._df,
        None,
    )
    node = MapFunction(
        schema,
        "hint_sorted",
        [[("a", descending, nulls_last)]],
        child,
    )

    result = node.evaluate(cache={}, timer=None, context=IRExecutionContext())

    order = plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING
    null_order = (
        plc.types.NullOrder.AFTER
        if descending != nulls_last
        else plc.types.NullOrder.BEFORE
    )
    assert result.column_map["a"].check_sorted(
        order=order, null_order=null_order, stream=result.stream
    )
    assert result.column_map["b"].is_sorted == plc.types.Sorted.NO


def test_hint_sorted_marks_multiple_column_metadata() -> None:
    schema = {
        "a": DataType(pl.Int64()),
        "b": DataType(pl.Int64()),
        "c": DataType(pl.Int64()),
    }
    child = DataFrameScan(
        schema,
        pl.DataFrame(
            {
                "a": [2, None, 1],
                "b": [2, None, 3],
                "c": [3, 1, 2],
            }
        )._df,
        None,
    )
    node = MapFunction(
        schema,
        "hint_sorted",
        [[("a", False, False), ("b", True, False)]],
        child,
    )

    result = node.evaluate(cache={}, timer=None, context=IRExecutionContext())

    assert result.column_map["a"].check_sorted(
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.BEFORE,
        stream=result.stream,
    )
    assert result.column_map["b"].check_sorted(
        order=plc.types.Order.DESCENDING,
        null_order=plc.types.NullOrder.AFTER,
        stream=result.stream,
    )
    assert result.column_map["c"].is_sorted == plc.types.Sorted.NO


def test_hint_sorted_normalized_options_roundtrip() -> None:
    schema = {"a": DataType(pl.Int64())}
    child = DataFrameScan(schema, pl.DataFrame({"a": [1]})._df, None)
    node = MapFunction(schema, "hint_sorted", [[("a", False, False)]], child)
    reconstructed = MapFunction(schema, "hint_sorted", node.options, child)

    assert reconstructed.options == node.options


def test_hint_sorted_streaming_lowering_preserves_partitioning() -> None:
    schema = {"a": DataType(pl.Int64())}
    child = DataFrameScan(schema, pl.DataFrame({"a": [1, 2, 3]})._df, None)
    node = MapFunction(schema, "hint_sorted", [[("a", False, False)]], child)
    child_partition = PartitionInfo(count=3)

    class Rec:
        @property
        def state(self) -> State:
            raise AssertionError("state is not used by hint_sorted lowering")

        def __call__(self, ir: IR) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
            assert ir is child
            return ir, {ir: child_partition}

    lowered, partition_info = lower_ir_node(node, Rec())

    assert isinstance(lowered, MapFunction)
    assert lowered.name == "hint_sorted"
    assert partition_info[lowered] is child_partition


def test_explode_single_legacy_options():
    # Cover the branch: POLARS_VERSION_LT_136 or len(self.options) == 1
    # On polars >= 1.36 this branch is only reachable by direct construction
    # with 1-element options (the old pre-1.36 format).
    df = pl.DataFrame({"a": [[1, 2], [3, 4]]})
    child = DataFrameScan({"a": DataType(pl.List(pl.Int64()))}, df._df, None)
    # 1-element options list: [[columns_to_explode]]
    node = MapFunction({"a": DataType(pl.Int64())}, "explode", [["a"]], child)
    assert node.options == (("a",),)
