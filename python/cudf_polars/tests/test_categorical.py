# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import weakref
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.translate import Translator
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.config import Cluster
from cudf_polars.utils.cuda_stream import get_cuda_stream
from cudf_polars.utils.versions import POLARS_VERSION_LT_136, POLARS_VERSION_LT_140

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def nulls() -> str:
    return "some"


@pytest.fixture(
    params=[
        pytest.param(pl.Categorical(), id="categorical_global"),
        pytest.param(pl.Categorical("fruit"), id="categorical_named"),
        pytest.param(
            pl.Categorical(pl.Categories("small", physical=pl.UInt8)),
            id="categorical_uint8",
        ),
        pytest.param(
            pl.Categorical(pl.Categories("mid", "ns", pl.UInt16)),
            id="categorical_uint16",
        ),
        pytest.param(pl.Enum(["a", "b", "c"]), id="enum"),
        pytest.param(pl.Enum([str(i) for i in range(300)]), id="enum_uint16"),
    ]
)
def categorical_frame(request: pytest.FixtureRequest, nulls: str) -> pl.LazyFrame:
    dtype = request.param
    categories = (
        dtype.categories.to_list() if isinstance(dtype, pl.Enum) else ["c", "a", "b"]
    )
    values: list[str | None]
    if nulls == "empty":
        values = []
    elif nulls == "all":
        values = [None] * 10
    else:
        values = [categories[(i * 7) % len(categories)] for i in range(10)]
        if nulls == "some":
            values = [None if i % 5 == 0 else v for i, v in enumerate(values)]
    return pl.LazyFrame(
        {
            "int": pl.Series(range(len(values)), dtype=pl.Int64),
            "cat": pl.Series(values, dtype=dtype),
            "other": pl.Series(values, dtype=dtype),
        }
    )


def assert_passthrough(q: pl.LazyFrame, engine: pl.GPUEngine, **kwargs: Any) -> None:
    if any(
        isinstance(dtype, pl.Categorical) for dtype in q.collect_schema().values()
    ) and engine.config.get("executor_options", {}).get("cluster") in {
        Cluster.RAY,
        Cluster.DASK,
    }:
        assert_ir_translation_raises(q, engine, NotImplementedError)
    else:
        assert_gpu_result_equal(q, engine=engine, **kwargs)


@pytest.mark.parametrize("nulls", ["none", "some", "all", "empty"])
def test_dataframe_roundtrip(categorical_frame):
    df = categorical_frame.collect()
    result = DataFrame.from_polars(df, stream=get_cuda_stream())

    for column in result.columns:
        assert column.obj.type() == column.dtype.plc_type
    physical = df.select(pl.all().to_physical())
    codes = DataFrame(
        [
            Column(c.obj, name=c.name, dtype=DataType(physical.schema[c.name]))
            for c in result.columns
        ],
        stream=result.stream,
    ).to_polars()
    assert_frame_equal(codes, physical)
    assert_frame_equal(result.to_polars(), df, check_dtypes=True)


def test_dataframe_roundtrip_empty_enum():
    df = pl.DataFrame({"a": pl.Series([None, None], dtype=pl.Enum([])), "b": [1, 2]})
    result = DataFrame.from_polars(df, stream=get_cuda_stream()).to_polars()
    assert_frame_equal(result, df, check_dtypes=True)


@pytest.mark.parametrize(
    "dtype, expected",
    [(pl.Categorical("sortedness"), False), (pl.Enum(["a", "b"]), True)],
    ids=["categorical", "enum"],
)
def test_to_polars_sortedness(dtype, expected):
    df = pl.DataFrame({"a": pl.Series(["a", "b"], dtype=dtype)})
    gpu_df = DataFrame.from_polars(df, stream=get_cuda_stream())
    gpu_df.column_map["a"].set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.BEFORE,
    )
    result = gpu_df.to_polars()

    assert_frame_equal(result, df, check_dtypes=True)
    assert result["a"].flags["SORTED_ASC"] is expected


@pytest.mark.parametrize("nulls", ["none", "some", "all", "empty"])
def test_roundtrip(categorical_frame, engine):
    assert_passthrough(categorical_frame, engine)


def test_roundtrip_empty_enum(engine):
    q = pl.LazyFrame({"a": pl.Series([None, None], dtype=pl.Enum([])), "b": [1, 2]})
    assert_gpu_result_equal(q, engine=engine)


def test_select(categorical_frame, engine):
    q = categorical_frame.select(
        pl.col("other"), pl.col("cat").alias("renamed"), pl.col("int")
    )
    assert_passthrough(q, engine)


def test_with_columns(categorical_frame, engine):
    q = categorical_frame.with_columns(pl.col("cat").alias("cat2"))
    assert_passthrough(q, engine)


def test_filter(categorical_frame, engine):
    q = categorical_frame.filter(pl.col("int") > 4)
    assert_passthrough(q, engine)


@pytest.mark.parametrize(
    "op",
    [
        lambda q: q.head(3),
        lambda q: q.tail(3),
        lambda q: q.slice(3, 5),
    ],
    ids=["head", "tail", "slice"],
)
def test_slice(categorical_frame, engine, op):
    assert_passthrough(op(categorical_frame), engine)


def test_sort_by_non_categorical(categorical_frame, engine):
    q = categorical_frame.sort("int", descending=True)
    assert_passthrough(q, engine)


@pytest.mark.parametrize("how", ["inner", "left", "full"])
@pytest.mark.parametrize("both_sides", [False, True], ids=["one_side", "both_sides"])
def test_join_payload(categorical_frame, engine, how, both_sides):
    right = categorical_frame.head(5)
    if not both_sides:
        right = right.select("int", pl.col("int").alias("right_value"))
    q = categorical_frame.join(right, on="int", how=how)
    assert_passthrough(q, engine, check_row_order=False)


def test_group_by_ignores_categorical(categorical_frame, engine):
    q = categorical_frame.with_columns(pl.col("int") % 3).group_by("int").agg(pl.len())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_group_by_with_categorical_payload_raises(categorical_frame, in_memory_engine):
    q = categorical_frame.group_by("int").agg(pl.col("cat"), pl.len())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_concat_vertical(categorical_frame, engine):
    q = pl.concat([categorical_frame, categorical_frame.head(3)])
    assert_passthrough(q, engine, check_row_order=False)


def test_concat_horizontal(categorical_frame, engine):
    right = categorical_frame.select(pl.col("cat").alias("cat_right"))
    q = pl.concat([categorical_frame, right], how="horizontal")
    assert_passthrough(q, engine)


def test_cache(categorical_frame, engine):
    lf = categorical_frame.filter(pl.col("int") > 5)
    q = pl.concat([lf, lf.select(pl.col("int"), pl.col("cat"), pl.col("other"))])
    assert_passthrough(q, engine, check_row_order=False)


def test_row_index(categorical_frame, engine):
    q = categorical_frame.head(3).with_row_index()
    assert_passthrough(q, engine)


def test_rename(categorical_frame, engine):
    q = categorical_frame.rename({"cat": "renamed"})
    assert_passthrough(q, engine)


def test_explode(categorical_frame, engine):
    q = (
        categorical_frame.collect()
        .with_columns(lst=pl.concat_list(pl.col("int"), pl.col("int")))
        .lazy()
        .explode("lst")
    )
    assert_passthrough(q, engine)


@pytest.mark.parametrize(
    "dtype",
    [pl.Enum(["a", "b", "c"]), pl.Enum([str(i) for i in range(300)])],
    ids=["enum", "enum_uint16"],
)
def test_enum_multi_partition_shuffle_join(dtype, streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(
            target_partition_size=1,
            max_rows_per_partition=4,
            broadcast_limit=1,
            raise_on_fail=True,
        )
    )
    categories = dtype.categories.to_list()
    left = pl.LazyFrame(
        {
            "int": range(20),
            "cat": pl.Series(
                [None if i % 5 == 0 else categories[i % 3] for i in range(20)],
                dtype=dtype,
            ),
        }
    )
    right = left.with_columns(pl.col("int") % 7)
    q = left.join(right, on="int", how="inner")
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("how", ["left", "full"])
def test_null_codes_are_valid_after_join(categorical_frame, in_memory_engine, how):
    right = categorical_frame.head(5)
    got = categorical_frame.join(right, on="int", how=how).collect(
        engine=in_memory_engine
    )
    buffer = io.BytesIO()
    got.write_ipc(buffer)
    buffer.seek(0)

    assert_frame_equal(pl.read_ipc(buffer), got)


def test_mapping_lifetime():
    df = pl.DataFrame(
        {
            "a": pl.Series(
                ["p", "q", None],
                dtype=pl.Categorical(pl.Categories("mapping_lifetime")),
            )
        }
    )
    expected = df.cast(pl.String)
    gpu_df = DataFrame.from_polars(df, stream=get_cuda_stream())
    df_ref = weakref.ref(df)
    del df
    assert df_ref() is None

    assert_frame_equal(gpu_df.to_polars().cast(pl.String), expected)


@pytest.mark.parametrize(
    "make_query",
    [
        pytest.param(lambda lf, _: lf.sort("cat"), id="sort"),
        pytest.param(lambda lf, _: lf.sort("int", "cat"), id="sort_multi_key"),
        pytest.param(
            lambda lf, _: lf.group_by("cat").agg(pl.col("int").sum()), id="group_by"
        ),
        pytest.param(lambda lf, other: lf.join(other, on="cat"), id="join"),
        pytest.param(
            lambda lf, other: lf.join(other, on=["int", "cat"]), id="join_multi_key"
        ),
        pytest.param(lambda lf, _: lf.unique(), id="unique"),
        pytest.param(lambda lf, _: lf.unique(subset=["cat"]), id="unique_subset"),
        pytest.param(lambda lf, _: lf.filter(pl.col("cat") == "a"), id="compare"),
        pytest.param(lambda lf, _: lf.select(pl.col("cat").cast(pl.String)), id="cast"),
        pytest.param(lambda lf, _: lf.select(pl.col("cat").is_null()), id="is_null"),
        pytest.param(
            lambda lf, _: lf.group_by("int").agg(pl.col("cat").first()), id="agg_first"
        ),
        pytest.param(
            lambda lf, _: lf.group_by("int").agg(pl.col("cat")), id="agg_implode"
        ),
        pytest.param(
            lambda lf, _: lf.select(pl.col("int").sum().over("cat")), id="over"
        ),
        pytest.param(lambda lf, _: lf.select(pl.col("cat").sort()), id="sort_expr"),
        pytest.param(
            lambda lf, _: lf.select(pl.col("int").sort_by("cat")), id="sort_by_expr"
        ),
        pytest.param(
            lambda lf, _: lf.unpivot(index="int", on=["cat", "other"]), id="unpivot"
        ),
        pytest.param(
            lambda lf, _: lf.unpivot(index="cat", on=["int"]), id="unpivot_index"
        ),
    ],
)
def test_unsupported_operations_raise(categorical_frame, in_memory_engine, make_query):
    other = categorical_frame.head(5)
    assert_ir_translation_raises(
        make_query(categorical_frame, other), in_memory_engine, NotImplementedError
    )


@pytest.mark.parametrize("nulls", ["none"])
@pytest.mark.parametrize(
    "categorical_frame", [pl.Enum(["a", "b", "c"])], ids=["enum"], indirect=True
)
def test_merge_sorted_on_categorical_raises(categorical_frame, in_memory_engine):
    lf = categorical_frame.sort("int")
    q = lf.merge_sorted(lf, key="cat")
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_cast_string_to_enum_raises(in_memory_engine):
    q = pl.LazyFrame({"a": ["x", "y"]}).select(pl.col("a").cast(pl.Enum(["x", "y"])))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_enum_literal_raises(in_memory_engine):
    q = pl.LazyFrame({"a": [1, 2]}).select(pl.lit("a", dtype=pl.Enum(["a"])))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_scan_parquet_raises(categorical_frame, in_memory_engine, tmp_path):
    path = tmp_path / "categorical.parquet"
    categorical_frame.collect().write_parquet(path)
    q = pl.scan_parquet(path)
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_sink_parquet_raises(categorical_frame, in_memory_engine, tmp_path):
    q = categorical_frame.sink_parquet(tmp_path / "categorical.parquet", lazy=True)
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "nested",
    [
        pl.List(pl.Categorical()),
        pl.Struct({"a": pl.Enum(["x", "y"])}),
        pl.Array(pl.Enum(["x", "y"]), 1),
    ],
    ids=["list_categorical", "struct_enum", "array_enum"],
)
def test_nested_categorical_raises(nested, in_memory_engine):
    q = pl.LazyFrame({"a": pl.Series([None, None], dtype=nested)})
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "categorical_frame", [pl.Enum(["a", "b", "c"])], ids=["enum"], indirect=True
)
@pytest.mark.skipif(
    POLARS_VERSION_LT_140 and not POLARS_VERSION_LT_136,
    reason="Fails for 1.36-1.39 (inclusive)",
)
def test_hint_sorted_raises(categorical_frame, in_memory_engine):
    q = categorical_frame.set_sorted("cat")
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "categorical_frame", [pl.Categorical()], ids=["categorical"], indirect=True
)
def test_categorical_translates_on_default_singleton_engine(categorical_frame):
    translator = Translator(
        categorical_frame._ldf.visit(), pl.GPUEngine(executor="streaming")
    )
    translator.translate_ir()
    assert translator.errors == []


def test_categorical_rejected_on_ray_engine(categorical_frame, ray_engine):
    assert_passthrough(categorical_frame, ray_engine)


def test_categorical_rejected_on_dask_engine(categorical_frame, dask_engine):
    assert_passthrough(categorical_frame, dask_engine)
