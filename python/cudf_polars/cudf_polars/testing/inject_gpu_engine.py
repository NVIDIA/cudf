# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running polars test suite using the GPU engine."""

from __future__ import annotations

import sqlite3
from functools import partialmethod
from typing import TYPE_CHECKING

import packaging.version
import pytest

import polars

from cudf_polars.utils.config import StreamingFallbackMode

if TYPE_CHECKING:
    from collections.abc import Mapping


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add plugin-specific options."""
    group = parser.getgroup(
        "cudf-polars", "Plugin to set GPU as default engine for polars tests"
    )
    group.addoption(
        "--inject-gpu-engine",
        action="store",
        default="in-memory",
        choices=("in-memory", "spmd"),
        help="Which GPU engine variant to inject globally.",
    )
    group.addoption(
        "--inject-gpu-engine-blocksize",
        action="store",
        default="default",
        choices=("default", "small"),
        help=(
            "Blocksize mode for the 'spmd' engine. Set to 'small' to run most "
            "tests with multiple partitions. Ignored for 'in-memory'."
        ),
    )
    group.addoption(
        "--inject-gpu-engine-raise-on-fail",
        action="store_true",
        help=(
            "Force raise_on_fail=True on the injected engine and suppress the "
            "plugin's xfail markers (tests will surface real failures)."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Enable use of this module as a pytest plugin to enable GPU collection."""
    variant = config.getoption("--inject-gpu-engine")
    blocksize = config.getoption("--inject-gpu-engine-blocksize")
    raise_on_fail = config.getoption("--inject-gpu-engine-raise-on-fail")

    if variant == "in-memory":
        engine = polars.GPUEngine(executor="in-memory", raise_on_fail=raise_on_fail)
    else:
        from cudf_polars.engine.spmd import SPMDEngine

        executor_options: dict[str, object] = {}
        if blocksize == "small":
            executor_options["max_rows_per_partition"] = 4
            executor_options["target_partition_size"] = 10
            # We expect many tests to fall back, so silence the warnings.
            executor_options["fallback_mode"] = StreamingFallbackMode.SILENT
        engine = SPMDEngine(
            executor_options=executor_options,
            engine_options={"raise_on_fail": raise_on_fail},
        )

    collect = polars.LazyFrame.collect
    # https://github.com/python/mypy/issues/2427
    polars.LazyFrame.collect = partialmethod(collect, engine=engine)  # type: ignore[method-assign, assignment]
    config._inject_gpu_engine = engine  # type: ignore[attr-defined]
    _verify_collect_patch(engine)

    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*GPU engine does not support streaming or background collection",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*Query execution with GPU not possible",
    )


def _verify_collect_patch(engine: object) -> None:
    """
    Raise if ``polars.LazyFrame.collect`` is not the partialmethod we installed.

    ``partialmethod`` is a descriptor — accessing ``polars.LazyFrame.collect``
    via the class invokes ``partialmethod.__get__`` and returns a plain
    function, so we inspect the raw entry in ``LazyFrame.__dict__`` to see
    the underlying descriptor.
    """
    raw = polars.LazyFrame.__dict__.get("collect")
    if not isinstance(raw, partialmethod):
        raise TypeError(
            f"polars.LazyFrame.collect patch failed: expected partialmethod, "
            f"got {type(raw).__name__}"
        )
    bound = raw.keywords.get("engine")
    if bound is None:
        raise RuntimeError(
            "polars.LazyFrame.collect is a partialmethod but has no "
            "'engine' keyword bound"
        )
    if bound is not engine:
        raise RuntimeError(
            f"polars.LazyFrame.collect has a different engine bound "
            f"({type(bound).__name__}) than the one we installed "
            f"({type(engine).__name__})"
        )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Tear down the injected engine."""
    engine = getattr(config, "_inject_gpu_engine", None)
    if engine is None:
        return
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        shutdown()


def pytest_report_header(config: pytest.Config) -> str:
    """Report which GPU engine has been injected into polars.LazyFrame.collect."""
    engine = getattr(config, "_inject_gpu_engine", None)
    cls = type(engine)
    return f"injected GPU engine: {cls.__module__}.{cls.__name__}"


EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/unit/io/test_csv.py::test_read_csv_only_loads_selected_columns": "Memory usage won't be correct due to GPU",
    "tests/unit/io/test_delta.py::test_scan_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_relative": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_read_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_schema_evolution_nested_struct_field_19915": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_nanosecond_timestamp": "polars generates the wrong schema: https://github.com/pola-rs/polars/issues/23949",
    "tests/unit/io/test_delta.py::test_scan_delta_nanosecond_timestamp_nested": "polars generates the wrong schema: https://github.com/pola-rs/polars/issues/23949",
    "tests/unit/io/test_iceberg.py::test_scan_iceberg_row_index_renamed": "Iceberg support not yet implemented in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_scan_iceberg_extra_columns": "Iceberg support not yet implemented in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_scan_iceberg_extra_struct_fields": "Iceberg support not yet implemented in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_scan_iceberg_column_deletion": "Iceberg schema evolution not yet implemented in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_scan_iceberg_nested_column_cast_deletion_rename": "Iceberg column_mapping (schema evolution) not yet implemented in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_scan_iceberg_parquet_prefilter_with_column_mapping": "Iceberg column_mapping (schema evolution) not yet implemented in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_fill_missing_fields_with_identity_partition_values_nested": "Iceberg partition column injection not yet implemented in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_scan_iceberg_fast_count[native]": "Iceberg fast count from metadata not yet supported in cudf-polars",
    "tests/unit/io/test_iceberg.py::test_iceberg_filter_bool_26474": "Iceberg support not yet implemented in cudf-polars",
    "tests/unit/io/test_io_plugin.py::test_defer_validate_false": "cudf-polars always validates the IO source schema, so validate_schema=False dtype mismatches are unsupported on GPU",
    "tests/unit/io/test_io_plugin.py::test_datetime_io_predicate_pushdown_21790": "cudf-polars validates the IO source schema exactly and does not coerce datetime time units (us vs ns)",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[small.parquet-4]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[foods*.parquet-54]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_is_in_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type0]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type1]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type2]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type3]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/test_sink.py::test_collect_all_lazy": "SinkMultiple not supported by InMemory CPU Engine, which we fallback to. See pola-rs/polars/pull/26537",
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249[False-False]": "Needs some variant of cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249[True-False]": "Needs some variant of cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_unaligned_schema_read[False]": "Incomplete handling of projected reads with mismatching schemas, cudf#16394",
    "tests/unit/io/test_parquet.py::test_read_parquet_only_loads_selected_columns_15098": "Memory usage won't be correct due to GPU",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_scan_parquet_filter_statistics_load_missing_column_21391": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_binary_offset_roundtrip": "binary offset type unsupported",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[gpu]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[engine1]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_lazyframe.py::test_round[dtype2-123.55-1-123.6]": "libcudf HALF_EVEN rounding bug for Float64 with decimal_places > 0. See https://github.com/rapidsai/cudf/issues/21319",
    "tests/unit/lazyframe/test_lazyframe.py::test_cast_frame": "Casting that raises not supported on GPU",
    "tests/unit/lazyframe/test_lazyframe.py::test_lazy_cache_hit": "Debug output on stderr doesn't match",
    "tests/unit/operations/aggregation/test_aggregations.py::test_nan_inf_aggregation": "treatment of nans and nulls together is different in libcudf and polars in groupby-agg context",
    "tests/unit/operations/test_abs.py::test_abs_duration": "Need to raise for unsupported uops on timelike values",
    "tests/unit/operations/test_group_by.py::test_group_by_shorthand_quantile": "libcudf quantiles are round to nearest ties to even, polars quantiles are round to nearest ties away from zero",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input10-expected10-Date-output_dtype10]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input11-expected11-input_dtype11-output_dtype11]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input12-expected12-input_dtype12-output_dtype12]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input13-expected13-input_dtype13-output_dtype13]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input14-expected14-input_dtype14-output_dtype14]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input15-expected15-input_dtype15-output_dtype15]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input16-expected16-input_dtype16-output_dtype16]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_grouped_slice_literals[False]": "List literal loses nesting in groupby-agg: cudf#19610",
    "tests/unit/operations/test_group_by.py::test_grouped_slice_literals[True]": "List literal loses nesting in groupby-agg: cudf#19610",
    "tests/unit/operations/test_group_by.py::test_group_broadcast_binary_apply_expr_25046[pow-rhs1-lhs0]": "libcudf integer pow uses float exp/log, so 3**1 rounds to 2",
    "tests/unit/operations/test_group_by.py::test_group_broadcast_binary_apply_expr_25046[pow-rhs2-lhs0]": "libcudf integer pow uses float exp/log, so 3**1 rounds to 2",
    "tests/unit/operations/test_group_by.py::test_group_broadcast_binary_apply_expr_25046[pow-rhs3-lhs0]": "libcudf integer pow uses float exp/log, so 3**1 rounds to 2",
    "tests/unit/operations/test_group_by.py::test_group_broadcast_binary_apply_expr_25046[pow-rhs4-lhs0]": "libcudf integer pow uses float exp/log, so 3**1 rounds to 2",
    "tests/unit/operations/test_group_by.py::test_group_by_binary_agg_with_literal": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_group_by.py::test_group_by_lit_series": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_join.py::test_cross_join_slice_pushdown": "Need to implement slice pushdown for cross joins",
    # We match the behavior of the polars[cpu] streaming engine (it makes doesn't make any ordering guarantees either when maintain_order is none).
    # But this test does because the test is run with the polars[cpu] in-memory engine, which still preserves the order of the left dataframe
    # when maintain order is none.
    "tests/unit/operations/test_join.py::test_join_preserve_order_left": "polars[gpu] makes no ordering guarantees when maintain_order is none",
    # TODO: As of polars 1.34, the column names for left and right came in unaligned, which causes the dtypes to mismatch when calling plc.replace.replace_nulls
    # Need to investigate what changed in polars
    "tests/unit/operations/test_join.py::test_join_coalesce_column_order_23177": "Misaligned left/right column names left and right tables in join op",
    "tests/unit/operations/namespaces/string/test_pad.py::test_str_zfill_unicode_not_respected": "polars doesn't add zeros for unicode characters.",
    "tests/unit/sql/test_cast.py::test_cast_errors[values0-values::uint8-conversion from `f64` to `u64` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values1-values::uint4-conversion from `i64` to `u32` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values2-values::int1-conversion from `i64` to `i8` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values5-values::int4-conversion from `str` to `i32` failed]": "Cast raises, but error user receives is wrong",
    "tests/unit/lazyframe/test_predicates.py::test_predicate_pushdown_split_pushable": "Casting that raises not supported on GPU",
    "tests/unit/lazyframe/test_predicates.py::test_filter_contradiction_fallible_error_handling": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_miscellaneous.py::test_read_csv": "Incorrect handling of missing_is_null in read_csv",
    "tests/unit/lazyframe/test_cse.py::test_cse_predicate_self_join[False]": "Debug output on stderr doesn't match, see https://github.com/rapidsai/cudf/issues/22967",
    "tests/unit/lazyframe/test_cse.py::test_cse_predicate_self_join[True]": "Debug output on stderr doesn't match, see https://github.com/rapidsai/cudf/issues/22967",
    "tests/unit/io/test_scan_row_deletion.py::test_scan_row_deletion_skips_file_with_all_rows_deleted": "The test intentionally corrupts the parquet file, so we cannot read the row count from the header.",
    "tests/unit/io/test_multiscan.py::test_multiscan_row_index[scan_csv-write_csv-csv]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-columns]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-row_groups]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-prefiltered]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-none]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-columns]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-row_groups]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-prefiltered]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-none]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_multiscan.py::test_multiscan_row_index[scan_csv-write_csv]": "CSV multiscan with row_index and no row limit is not yet supported.",
    "tests/unit/functions/test_concat.py::test_concat_with_empty_dataframes_strict_25725": "https://github.com/rapidsai/cudf/issues/21644",
    "tests/unit/sql/test_window_functions.py::test_over_with_cumulative_window_funcs": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/sql/test_window_functions.py::test_window_cumulative_agg_with_nulls": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/sql/test_window_functions.py::test_window_named_window": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/operations/test_window.py::test_over_literal_cum_sum_26800": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/operations/namespaces/array/test_array.py::test_array_idx_size_limit_eval": "polars-internal IdxSize chunking debug assertion does not apply with the GPU engine",
    "tests/unit/operations/aggregation/test_aggregations.py::test_implode_and_agg": "implode + agg returns a mismatched dtype",
    "tests/unit/operations/aggregation/test_aggregations.py::test_duration_aggs": "Unsupported libcudf reduction operator for Duration dtype",
    "tests/unit/operations/aggregation/test_aggregations.py::test_boolean_aggs": "boolean-agg mean floating-point precision mismatch",
    "tests/unit/io/test_scan.py::test_scan_sink_metrics_multiple_phases": "sink metrics are not reported by the GPU engine",
    "tests/unit/io/test_parquet.py::test_read_parquet_legacy_nested_maps_27159": "legacy nested-map parquet read produces a mismatched result",
    "tests/unit/datatypes/test_struct.py::test_struct_equal_missing_null_25360": "struct equality with a null raises libcudf 'Index out of bounds' (get_element)",
}


TESTS_TO_SKIP: dict[str, str] = {
    "tests/unit/operations/test_profile.py::test_profile_with_cse": "Shape assertion won't match",
    # value_counts / struct-expansion row ordering is not guaranteed, so the GPU
    # result may or may not match CPU. Skip rather than xfail to avoid a flaky
    # XPASS/FAIL (these pass on some runs and fail on others).
    "tests/unit/lazyframe/test_cse.py::test_cse_as_struct_value_counts_20927": "non-deterministic value_counts ordering",
    "tests/unit/lazyframe/test_cse.py::test_eager_cse_during_struct_expansion_18411": "non-deterministic struct-expansion ordering",
    # On Ubuntu 20.04, the tzdata package contains a bunch of symlinks
    # for obsolete timezone names. However, the chrono_tz package that
    # polars uses doesn't read /usr/share/zoneinfo, instead packaging
    # the current zoneinfo database from IANA. Consequently, when this
    # hypothesis-generated test runs and generates timezones from the
    # available zoneinfo-reported timezones, we can get an error from
    # polars that the requested timezone is unknown.
    # Since this is random, just skip it, rather than xfailing.
    "tests/unit/lazyframe/test_serde.py::test_lf_serde_roundtrip_binary": "chrono_tz doesn't have all tzdata symlink names",
    # Tests performance difference of CPU engine
    "tests/unit/operations/test_join.py::test_join_where_eager_perf_21145": "Tests performance bug in CPU engine",
    "tests/unit/operations/namespaces/list/test_list.py::test_list_struct_field_perf": "Tests CPU Engine perf",
    "tests/benchmark/test_with_columns.py::test_with_columns_quadratic_19503": "Tests performance bug in CPU engine",
    # The test may segfault with the legacy streaming engine. We should
    # remove this skip when all polars tests use the new streaming engine.
    # Fails in CI, but passes locally
    "tests/unit/streaming/test_streaming.py::test_streaming_streamable_functions": "RuntimeError: polars_python::sql::PySQLContext is unsendable, but is being dropped on another thread",
    # Remove when polars supports Pydantic V3
    "tests/unit/constructors/test_constructors.py::test_init_structured_objects": "pydantic deprecation warning",
    "tests/unit/constructors/test_constructors.py::test_init_pydantic_2x": "pydantic deprecation warning",
    "tests/unit/constructors/test_constructors.py::test_init_structured_objects_nested[_TestFooPD-_TestBarPD-_TestBazPD]": "pydantic deprecation warning",
    "tests/unit/series/test_series.py::test_init_structured_objects": "pydantic deprecation warning",
    "tests/unit/series/test_describe.py::test_series_describe_float": "https://github.com/rapidsai/cudf/issues/19324",
    "tests/unit/series/test_describe.py::test_series_describe_int": "https://github.com/rapidsai/cudf/issues/19324",
    "tests/unit/streaming/test_streaming.py::test_streaming_apply": "https://github.com/pola-rs/polars/issues/22558",
    "tests/unit/operations/test_group_by_dynamic.py::test_group_by_dynamic_agg_bad_input_types[str]": "TODO: Need to investigate why this fails in CI but passes locally. We should fallback to CPU for group_by_dynamic",
    "tests/unit/expr/test_exprs.py::test_exp_log1p[Float16-Float16]": "Flaky test: Small floating-point precision differences in exp/log1p results",
    # TODO: Investigate why these tests fail in CI but pass locally.
    "tests/unit/io/test_delta.py::test_scan_delta_extract_table_statistics_df": "schemas mismatch: dtypes different",
    "tests/unit/io/test_partition.py::test_sink_partitioned_no_columns_in_file_25535[scan_parquet-sink_parquet]": "Incorrect row count. Related to https://github.com/rapidsai/cudf/issues/21428",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[0]": "ZeroDivisionError: division by zero",
    # Flaky deadlock test, may occur on rtxpro6000 only
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_in_mem_to_streaming_dispatch_deadlock_22641": "Flaky deadlock, may occur on rtxpro6000 only",
    # Short term adds in the aftermath of the rapidsmpf switch to get CI passing
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_local_with_async": "Flaky, otherwise TBD",
    "tests/unit/operations/test_join.py::test_join_where_nested_expr_21066": "Flaky, otherwise TBD",
    "tests/unit/io/test_scan.py::test_scan_metrics[True-parquet]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[True-csv]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[True-ndjson]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[False-parquet]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[False-csv]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[False-ndjson]": "Checks to IO metric logs specific to Polars CPU",
    # polars 1.42 updated these tests to also assert deprecated_call and strict=True ShapeError
    # in the same test function. The SPMD engine fails on the strict=True collect() inside a
    # pytest.raises block because the DeprecationWarning from how='horizontal' propagates differently
    # across engines. Skip both runs rather than xfail (which would XPASS on in-memory).
    "tests/unit/lazyframe/test_predicates.py::test_hconcat_predicate": "polars 1.42: test uses deprecated how='horizontal' with strict=True in ways that behave differently across GPU engines",
    "tests/unit/functions/test_union.py::test_union_lazyframe_horizontal": "polars 1.42: test uses deprecated how='horizontal' with strict=True in ways that behave differently across GPU engines",
}


if packaging.version.parse(sqlite3.sqlite_version) <= packaging.version.parse("3.44.0"):
    # These tests rely on features not available in older versions of sqlite.
    TESTS_TO_SKIP.update(
        {
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[SUM(x) FILTER (WHERE y > 20)-values0]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[AVG(x) FILTER (WHERE y > 20)-values1]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[MIN(x) FILTER (WHERE grp = 'a')-values2]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[MAX(x) FILTER (WHERE grp = 'a')-values3]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[COUNT(*) FILTER (WHERE grp = 'a')-values4]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[COUNT(1) FILTER (WHERE grp = 'a')-values5]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[COUNT(x) FILTER (WHERE grp = 'a')-values6]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[COUNT(x) FILTER (WHERE y > 20)-values7]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_grouped[COUNT(DISTINCT x) FILTER (WHERE y > 20)-values8]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_no_group_by[SUM(x) FILTER (WHERE y > 20)-13]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_no_group_by[AVG(x) FILTER (WHERE y > 20)-4.333333333333333]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_no_group_by[COUNT(*) FILTER (WHERE grp = 'a')-3]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_no_group_by[COUNT(x) FILTER (WHERE y > 20)-3]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_no_group_by[COUNT(DISTINCT x) FILTER (WHERE grp = 'b')-3]": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_filter_clause.py::test_filter_clause_multiple_aggs": 'sqlite3.OperationalError: near "AS": syntax error',
            "tests/unit/sql/test_string_agg.py::test_string_agg_aliases[STRING_AGG]": 'sqlite3.OperationalError: near "ORDER": syntax error',
            "tests/unit/sql/test_string_agg.py::test_string_agg_aliases[GROUP_CONCAT]": 'sqlite3.OperationalError: near "ORDER": syntax error',
        }
    )


# Generally skip for:
# 1) Tests that are too slow with --inject-gpu-engine-blocksize=small due to many small partitions for large data
STREAMING_ENGINE_TESTS_TO_SKIP: Mapping[str, str] = {
    "tests/unit/operations/aggregation/test_aggregations.py::test_boolean_aggs": "float difference in std/var in the unit of least precision",
    # No deterministic key sort (https://github.com/rapidsai/cudf/issues/21641):
    # passes on some streaming runs and fails on others, so skip rather than
    # xfail to avoid a flaky XPASS/FAIL.
    "tests/unit/operations/test_group_by.py::test_group_by_unique_parametric[n_unique-True-True]": "non-deterministic key sort under the streaming engine",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q1": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q2": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q3": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q4": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q5": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q7": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q10": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_join_where.py::test_single_inequality": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_join_where.py::test_non_strict_inequalities": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/benchmark/test_join_where.py::test_strict_inequalities": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_partition.py::test_partition_approximate_size": "Too slow for CI",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_many_row_groups_12297": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan[single-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan[single-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-parquet-async]": "Takes >60 seconds to run locally",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[glob-parquet-async]": "Takes >60 seconds to run locally",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[glob-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[single-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[single-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs0-True-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs1-True-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs2-True-unordered_columns2]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs3-True-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs4-True-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs5-True-unordered_columns5]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs6-False-unordered_columns6]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs7-False-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs8-False-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs9-True-unordered_columns9]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs10-True-unordered_columns10]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs11-False-unordered_columns11]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs12-False-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_order_observability.py::test_with_columns_sensitivity[exprs13-False-None]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_optimizations.py::test_collapse_joins_combinations": "Too slow for CI",
    "tests/unit/operations/test_index_of.py::test_randomized": "Too slow for CI; marked as pytest.mark.slow",
    "tests/unit/operations/test_slice.py::test_slice_slice_pushdown": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Categorical-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Categorical-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-1056-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-1056-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-1056-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_overflow_mean_partitioned_group_by_5194[Int32]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_overflow_mean_partitioned_group_by_5194[UInt32]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_sort.py::test_streaming_sort_varying_order_and_dtypes[sort_by0]": "Too slow for CI",
}

# xfail for tests that produce different results than CPU Polars
STREAMING_ENGINE_EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/unit/functions/range/test_linear_space.py::test_linear_space_num_samples_expr": "https://github.com/rapidsai/cudf/issues/22072",
    "tests/unit/functions/test_concat.py::test_concat_horizontal_zero_width_height_mismatch_26876": "https://github.com/rapidsai/cudf/issues/21644",
    "tests/unit/functions/test_concat.py::test_concat_horizontally_strict": "Correct polars.exceptions.ShapeError raised but it's in a ExceptionGroup",
    "tests/unit/functions/test_when_then.py::test_mismatched_height_should_raise[ternary_expr0-df0]": "Correct polars.exceptions.ShapeError raised but it's in a ExceptionGroup",
    "tests/unit/functions/test_when_then.py::test_mismatched_height_should_raise[ternary_expr0-df1]": "Correct polars.exceptions.ShapeError raised but it's in a ExceptionGroup",
    "tests/unit/functions/test_when_then.py::test_mismatched_height_should_raise[ternary_expr1-df0]": "Correct polars.exceptions.ShapeError raised but it's in a ExceptionGroup",
    "tests/unit/functions/test_when_then.py::test_mismatched_height_should_raise[ternary_expr1-df1]": "Correct polars.exceptions.ShapeError raised but it's in a ExceptionGroup",
    "tests/unit/operations/test_slice.py::test_slice_pushdown_literal_projection_14349": "https://github.com/rapidsai/cudf/issues/22072",
    "tests/unit/operations/test_group_by.py::test_group_by_lit_series": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_group_by.py::test_group_by_series_partitioned": "https://github.com/rapidsai/cudf/issues/22072",
    "tests/unit/operations/test_group_by.py::test_partitioned_group_by_chunked": "https://github.com/rapidsai/cudf/issues/22072",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[1]": "https://github.com/rapidsai/cudf/issues/22075",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[4]": "https://github.com/rapidsai/cudf/issues/22075",
    "tests/unit/operations/aggregation/test_aggregations.py::test_item_too_many": "Correct polars.exceptions.ComputeError raised but it's in an ExceptionGroup",
    "tests/unit/operations/aggregation/test_aggregations.py::test_single_empty": "Correct polars.exceptions.ComputeError raised but it's in an ExceptionGroup",
    "tests/unit/operations/test_join.py::test_empty_outer_join_22206": "https://github.com/rapidsai/cudf/issues/22084",
    "tests/unit/operations/test_replace.py::test_replace_invalid_old_dtype": "Correct InvalidOperationError raised but it's in an ExceptionGroup",
    "tests/unit/operations/test_window.py::test_over_literal_cum_sum_26800": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/sql/test_joins.py::test_cross_join_unnest_from_cte": "https://github.com/rapidsai/cudf/issues/22073",
    "tests/unit/sql/test_window_functions.py::test_over_with_cumulative_window_funcs": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/sql/test_window_functions.py::test_window_cumulative_agg_with_nulls": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/sql/test_window_functions.py::test_window_multiple_named_window": "TODO: https://github.com/rapidsai/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/functions/test_concat.py::test_concat_horizontal_lazy_strict_raises_shape_error_27415": "horizontal-concat strict height-mismatch raised inside an ExceptionGroup under the streaming engine",
    "tests/unit/io/test_io_plugin.py::test_defer_validate_true": "correct SchemaError raised but wrapped in an ExceptionGroup under the streaming engine",
    "tests/unit/operations/test_slice.py::test_hconcat_tail_unequal_heights_strict_raises_27552": "horizontal-concat strict height-mismatch raised inside an ExceptionGroup under the streaming engine",
}


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark known failing tests."""
    if config.getoption("--inject-gpu-engine-raise-on-fail"):
        # Don't xfail tests if running without fallback
        return
    with_streaming_engine = config.getoption("--inject-gpu-engine") == "spmd"
    for item in items:
        if (reason := TESTS_TO_SKIP.get(item.nodeid)) is not None or (
            with_streaming_engine
            and (reason := STREAMING_ENGINE_TESTS_TO_SKIP.get(item.nodeid, None))
            is not None
        ):
            item.add_marker(pytest.mark.skip(reason=reason))
        elif (
            with_streaming_engine
            and (s_reason := STREAMING_ENGINE_EXPECTED_FAILURES.get(item.nodeid, None))
            is not None
        ):
            item.add_marker(pytest.mark.xfail(reason=s_reason))
        elif (reason := EXPECTED_FAILURES.get(item.nodeid, None)) is not None:
            item.add_marker(pytest.mark.xfail(reason=reason))
