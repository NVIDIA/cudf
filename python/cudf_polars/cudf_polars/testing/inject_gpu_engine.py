# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running polars test suite using the GPU engine."""

from __future__ import annotations

import hashlib
import sqlite3
from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING

import packaging.version
import pytest

import polars

from cudf_polars.testing.engine_utils import (
    SMALL_MAX_ROWS_PER_PARTITION,
    SMALL_TARGET_PARTITION_SIZE,
)
from cudf_polars.testing.fallback import fallback_used, gpu_attempted, gpu_executed
from cudf_polars.testing.fallback_report import FallbackReport
from cudf_polars.utils.config import StreamingFallbackMode

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from pluggy import Result


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer pytest option."""
    integer = int(value)
    if integer < 0:
        raise ValueError(f"Argument {value} must be non-negative")
    return integer


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
            "Blocksize mode for the 'spmd' engine. 'small' forces most tests "
            "onto the multi-partition path. PR CI uses 'default'; nightly uses "
            "'small'. Ignored for 'in-memory'."
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
    group.addoption(
        "--inject-gpu-engine-report",
        metavar="PATH",
        help="Write per-test GPU execution and fallback diagnostics as JSON on the pytest controller.",
    )
    group.addoption(
        "--cudf-polars-shard-id",
        dest="cudf_polars_shard_id",
        type=nonnegative_int,
        default=0,
        help="Zero-based index of this cudf-polars test shard.",
    )
    group.addoption(
        "--cudf-polars-num-shards",
        dest="cudf_polars_num_shards",
        type=nonnegative_int,
        default=1,
        help="Total number of cudf-polars test shards.",
    )


def _sha256_hash(value: str) -> int:
    """Return a stable integer hash for a pytest node ID."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest(), "little")


def partition_items_by_shard(
    items: list[pytest.Item], shard_id: int, num_shards: int
) -> tuple[list[pytest.Item], list[pytest.Item]]:
    """Split items into this shard and items assigned to other shards."""
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        target = (
            selected
            if _sha256_hash(item.nodeid) % num_shards == shard_id
            else deselected
        )
        target.append(item)
    return selected, deselected


def validate_shard_options(shard_id: int, num_shards: int) -> None:
    """Validate cudf-polars test sharding options."""
    if num_shards < 1:
        raise pytest.UsageError(
            f"--cudf-polars-num-shards ({num_shards}) must be at least 1"
        )
    if not 0 <= shard_id < num_shards:
        raise pytest.UsageError(
            "--cudf-polars-shard-id "
            f"({shard_id}) must be in "
            f"[0, --cudf-polars-num-shards ({num_shards}))"
        )


def pytest_report_collectionfinish(
    config: pytest.Config, items: list[pytest.Item]
) -> str | None:
    """Report the number of tests assigned to a non-default shard."""
    num_shards = config.getoption("cudf_polars_num_shards")
    if num_shards > 1:
        shard_id = config.getoption("cudf_polars_shard_id")
        return f"Running {len(items)} items in shard {shard_id}/{num_shards}"
    return None


def pytest_configure(config: pytest.Config) -> None:
    """Enable use of this module as a pytest plugin to enable GPU collection."""
    variant = config.getoption("--inject-gpu-engine")
    blocksize = config.getoption("--inject-gpu-engine-blocksize")
    raise_on_fail = config.getoption("--inject-gpu-engine-raise-on-fail")
    validate_shard_options(
        config.getoption("cudf_polars_shard_id"),
        config.getoption("cudf_polars_num_shards"),
    )
    report_path = config.getoption("--inject-gpu-engine-report")
    # Workers send TestReport objects to the controller; only the controller
    # should aggregate them and write the shared output file.
    if report_path and not hasattr(config, "workerinput"):
        config.pluginmanager.register(
            FallbackReport(Path(report_path).resolve()), "cudf-polars-fallback-report"
        )

    if variant == "in-memory":
        engine = polars.GPUEngine(executor="in-memory", raise_on_fail=raise_on_fail)
    else:
        from cudf_polars.engine.spmd import SPMDEngine

        executor_options: dict[str, object] = {}
        if blocksize == "small":
            executor_options["max_rows_per_partition"] = SMALL_MAX_ROWS_PER_PARTITION
            executor_options["target_partition_size"] = SMALL_TARGET_PARTITION_SIZE
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


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, None, None]:
    """Give each injected-engine pytest item isolated execution signals."""
    signals = (fallback_used, gpu_attempted, gpu_executed)
    tokens = [signal.set(False) for signal in signals]
    try:
        yield
    finally:
        for signal, token in zip(signals, tokens, strict=True):
            signal.reset(token)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Result[pytest.TestReport], None]:
    """Attach per-test execution telemetry to each phase's report."""
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("--inject-gpu-engine-report"):
        # user_properties travel with TestReport across xdist's worker boundary.
        report.user_properties.extend(
            (name, str(signal.get()).lower())
            for name, signal in (
                ("cudf_polars_fallback", fallback_used),
                ("cudf_polars_gpu_attempted", gpu_attempted),
                ("cudf_polars_gpu_executed", gpu_executed),
            )
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


EXPECTED_FAILURES: dict[str, str] = {
    "tests/unit/io/test_io_plugin.py::test_datetime_io_predicate_pushdown_21790": "cudf-polars validates the IO source schema exactly and does not coerce datetime time units (us vs ns)",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[small.parquet-4]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[foods*.parquet-54]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_is_in_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type0]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type1]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type2]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type3]": "chunked-reader + include_file_paths bug: chunk.num_rows_per_source only reflects the first chunk",
    "tests/unit/io/cloud/test_cloud.py::test_sink_single_put_28356[sink_parquet-scan_parquet]": "cudf-polars does not forward storage_options to KvikIO, which rejects the s3:// URL without AWS environment variables",
    "tests/unit/io/cloud/test_metadata_prefetch.py::test_scan_parquet_metadata_without_head_request": "cudf-polars does not forward storage_options to KvikIO, which rejects the s3:// URL without AWS environment variables",
    "tests/unit/io/cloud/test_metadata_prefetch.py::test_scan_parquet_metadata_prefetch_too_small": "cudf-polars does not forward storage_options to KvikIO, which rejects the s3:// URL without AWS environment variables",
    "tests/unit/io/test_sink.py::test_collect_all_lazy": "SinkMultiple not supported by InMemory CPU Engine, which we fallback to. See pola-rs/polars/pull/26537",
    "tests/unit/io/test_parquet.py::test_resolve_heavy_sources_survives_an_unreadable_footer": "libcudf raises RuntimeError rather than polars.exceptions.ComputeError for a corrupted parquet footer",
    "tests/unit/io/test_parquet.py::test_parquet_known_source_sizes": "cudf-polars ignores the _source_sizes hint, so a wrong size does not raise polars.exceptions.ComputeError",
    "tests/unit/io/test_python_dataset.py::test_python_dataset_survives_an_unreadable_heavy_footer": "libcudf raises RuntimeError rather than polars.exceptions.ComputeError for a corrupted parquet footer",
    "tests/unit/io/test_lazy_parquet.py::test_statistics_of_rescaled_arrow_types[arrow_type3-<lambda>]": "Parquet filter pushdown does not cast literals to the file's duration unit",
    "tests/unit/io/test_lazy_parquet.py::test_statistics_of_rescaled_arrow_types[arrow_type4-<lambda>]": "Parquet filter pushdown does not cast literals to the file's date64 column type",
    "tests/unit/io/test_parquet.py::test_binary_offset_roundtrip": "binary offset type unsupported",
    "tests/unit/io/test_lazy_csv.py::test_scan_csv_schema_new_columns_dtypes[foods1.csv]": "CSV reader's use_cols_names picks the wrong columns when combined with a column_names_overwrite rename",
    "tests/unit/io/test_lazy_csv.py::test_scan_csv_schema_new_columns_dtypes[foods*.csv]": "CSV reader's use_cols_names picks the wrong columns when combined with a column_names_overwrite rename",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[gpu]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[engine1]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_engine_selection.py::test_default_engine_and_affinity[streaming]": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so the default engine and POLARS_ENGINE_AFFINITY are never consulted",
    "tests/unit/lazyframe/test_engine_selection.py::test_default_engine_and_affinity[None]": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so the default engine and POLARS_ENGINE_AFFINITY are never consulted",
    "tests/unit/cloud/test_remote_engine.py::test_assert_frame_equal_honors_remote_affinity": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so Config.engine_affinity's remote engine is never reached",
    "tests/unit/cloud/test_remote_engine.py::test_describe_honors_remote_affinity": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so Config.engine_affinity's remote engine is never reached",
    "tests/unit/lazyframe/test_engine.py::test_object_engine_affinity_drives_collect": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so Config.set_engine_affinity's custom engine is never reached",
    "tests/unit/lazyframe/test_query_monitoring.py::test_config_enable_monitoring": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so Config.enable_monitoring's streaming-affinity/observer hooks don't see the expected call counts",
    "tests/unit/lazyframe/test_query_monitoring.py::test_config_scope_monitoring": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so Config.enable_monitoring's streaming-affinity/observer hooks don't see the expected call counts",
    "tests/unit/lazyframe/test_query_monitoring.py::test_engine_affinity_object_carries_monitoring": "This plugin forces engine=<injected GPU engine> on LazyFrame.collect via partialmethod, so Config.engine_affinity's streaming-affinity/observer hooks don't see the expected call counts",
    "tests/unit/lazyframe/test_projections.py::test_projection_pushdown_union_len_pushdown_28657[concat]": "https://github.com/NVIDIA/cudf/issues/24112",
    "tests/unit/lazyframe/test_projections.py::test_projection_pushdown_union_len_pushdown_28657[union]": "https://github.com/NVIDIA/cudf/issues/24112",
    "tests/unit/lazyframe/test_lazyframe.py::test_round[dtype2-123.55-1-123.6]": "libcudf HALF_EVEN rounding bug for Float64 with decimal_places > 0. See https://github.com/NVIDIA/cudf/issues/21319",
    "tests/unit/lazyframe/test_lazyframe.py::test_cast_frame": "Casting that raises not supported on GPU",
    "tests/unit/lazyframe/test_lazyframe.py::test_lazy_cache_hit": "Debug output on stderr doesn't match",
    "tests/unit/operations/aggregation/test_aggregations.py::test_binary_op_agg_context_no_simplify_expr_12423": "groupby-agg of just literals should not produce collect_list",
    "tests/unit/operations/test_abs.py::test_abs_duration": "Need to raise for unsupported uops on timelike values",
    "tests/unit/operations/test_comparison.py::test_comparison_with_lower_precision_literal_matches_column[Datetime-ns-us]": "polars passes Datetime(ns) literals to the visitor as Python datetimes, which drops sub-microsecond precision",
    "tests/unit/operations/test_comparison.py::test_comparison_with_lower_precision_literal_matches_column[Datetime-ns-ms]": "polars passes Datetime(ns) literals to the visitor as Python datetimes, which drops sub-microsecond precision",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input10-expected10-Date-output_dtype10]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input11-expected11-input_dtype11-output_dtype11]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input12-expected12-input_dtype12-output_dtype12]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input13-expected13-input_dtype13-output_dtype13]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input14-expected14-input_dtype14-output_dtype14]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input15-expected15-input_dtype15-output_dtype15]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input16-expected16-input_dtype16-output_dtype16]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_lit_series": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_join.py::test_cross_join_slice_pushdown": "Need to implement slice pushdown for cross joins",
    # TODO: As of polars 1.34, the column names for left and right came in unaligned, which causes the dtypes to mismatch when calling plc.replace.replace_nulls
    # Need to investigate what changed in polars
    "tests/unit/operations/test_join.py::test_join_coalesce_column_order_23177": "Misaligned left/right column names left and right tables in join op",
    "tests/unit/operations/test_inequality_join.py::test_join_where_how_left": "join_where(how='left'/'right') is translated through the equi-join path using the predicate's columns as literal equality keys, ignoring the actual inequality condition",
    "tests/unit/operations/test_inequality_join.py::test_join_where_how_right": "join_where(how='left'/'right') is translated through the equi-join path using the predicate's columns as literal equality keys, ignoring the actual inequality condition",
    "tests/unit/sql/test_cast.py::test_cast_errors[values0-values::uint8-InvalidOperationError-conversion from `f64` to `u64` failed]": "Strict cast of out-of-range floats to integers does not raise",
    "tests/unit/sql/test_cast.py::test_cast_errors[values1-values::uint4-InvalidOperationError-conversion from `i64` to `u32` failed]": "Strict cast of out-of-range integers does not raise",
    "tests/unit/sql/test_cast.py::test_cast_errors[values2-values::int1-InvalidOperationError-conversion from `i64` to `i8` failed]": "Strict cast of out-of-range integers does not raise",
    "tests/unit/sql/test_cast.py::test_cast_errors[values4-values::int4-InvalidOperationError-conversion from `str` to `i32` failed]": "Strict cast failure message differs from polars",
    "tests/unit/sql/test_joins.py::test_constant_key_join_keeps_validation": "Join validation is not supported",
    "tests/unit/sql/test_joins.py::test_non_equi_left_join_null_keys": "SQL LEFT/RIGHT JOIN with an inequality condition is translated through the equi-join path, ignoring the inequality condition",
    "tests/unit/sql/test_joins.py::test_non_equi_right_join_pure": "SQL LEFT/RIGHT JOIN with an inequality condition is translated through the equi-join path, ignoring the inequality condition",
    "tests/unit/operations/namespaces/string/test_pad.py::test_str_zfill_unicode_not_respected": "polars doesn't add zeros for unicode characters.",
    "tests/unit/lazyframe/test_predicates.py::test_predicate_pushdown_split_pushable": "Casting that raises not supported on GPU",
    "tests/unit/lazyframe/test_predicates.py::test_filter_contradiction_fallible_error_handling": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_miscellaneous.py::test_read_csv": "Incorrect handling of missing_is_null in read_csv",
    "tests/unit/lazyframe/test_cse.py::test_cse_predicate_self_join[False]": "Debug output on stderr doesn't match, see https://github.com/NVIDIA/cudf/issues/22967",
    "tests/unit/lazyframe/test_cse.py::test_cse_predicate_self_join[True]": "Debug output on stderr doesn't match, see https://github.com/NVIDIA/cudf/issues/22967",
    "tests/unit/io/test_scan_row_deletion.py::test_scan_row_deletion_skips_file_with_all_rows_deleted": "The test intentionally corrupts the parquet file, so we cannot read the row count from the header.",
    "tests/unit/io/test_multiscan.py::test_multiscan_row_index[scan_csv-write_csv]": "CSV multiscan with row_index and no row limit is not yet supported.",
    "tests/unit/functions/test_concat.py::test_concat_with_empty_dataframes_strict_25725": "https://github.com/NVIDIA/cudf/issues/21644",
    "tests/unit/operations/test_window.py::test_over_literal_cum_sum_26800": "TODO: https://github.com/NVIDIA/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/operations/namespaces/array/test_array.py::test_array_idx_size_limit_eval": "polars-internal IdxSize chunking debug assertion does not apply with the GPU engine",
    "tests/unit/io/test_scan.py::test_scan_sink_metrics_multiple_phases": "sink metrics are not reported by the GPU engine",
    "tests/unit/io/test_parquet.py::test_read_parquet_concatenated_gzip_members_28787": "Non-deterministic garbage in the last row; likely a libcudf GZIP multi-member decompression bug",
    "tests/unit/datatypes/test_struct.py::test_struct_equal_missing_null_25360": "struct equality with a null raises libcudf 'Index out of bounds' (get_element)",
}


TESTS_TO_SKIP: dict[str, str] = {
    "tests/unit/io/test_delta.py::test_scan_delta_loads_aws_profile_endpoint_url": "Does not use the GPU engine; fails whenever AWS_DEFAULT_REGION is set, as in CI",
    # CI has shown two different failure modes across runs (a PanicException on
    # an invalid datetime, and garbage/uninitialized-looking datetime values),
    # suggesting a real non-deterministic bug in CSV datetime fallback parsing
    # under ignore_errors=True. Locally this hangs indefinitely instead of
    # reproducing either failure, so it can't be xfailed; skip instead.
    "tests/unit/io/test_csv.py::test_csv_datetime_fallback_contract[chunk-size-default]": "Non-deterministic CSV datetime fallback parsing; hangs locally instead of reproducing",
    "tests/unit/io/test_csv.py::test_csv_datetime_fallback_contract[chunk-size-7]": "Non-deterministic CSV datetime fallback parsing; hangs locally instead of reproducing",
    # value_counts / struct-expansion row ordering is not guaranteed, so the GPU
    # result may or may not match CPU. Skip rather than xfail to avoid a flaky
    # XPASS/FAIL (these pass on some runs and fail on others).
    "tests/unit/lazyframe/test_cse.py::test_cse_as_struct_value_counts_20927": "non-deterministic value_counts ordering",
    "tests/unit/lazyframe/test_cse.py::test_eager_cse_during_struct_expansion_18411": "non-deterministic struct-expansion ordering",
    # polars[gpu] makes no ordering guarantees when maintain_order is none, so
    # whether the GPU result matches the left dataframe order is an artifact of
    # the current join algorithm and may change. Skip rather than xfail to avoid
    # a flaky XPASS/FAIL.
    "tests/unit/operations/test_join.py::test_join_preserve_order_left": "join makes no ordering guarantees when maintain_order is none",
    # On Ubuntu 20.04, the tzdata package contains a bunch of symlinks
    # for obsolete timezone names. However, the chrono_tz package that
    # polars uses doesn't read /usr/share/zoneinfo, instead packaging
    # the current zoneinfo database from IANA. Consequently, when this
    # hypothesis-generated test runs and generates timezones from the
    # available zoneinfo-reported timezones, we can get an error from
    # polars that the requested timezone is unknown.
    # Since this is random, just skip it, rather than xfailing.
    "tests/unit/lazyframe/test_serde.py::test_lf_serde_roundtrip_binary": "chrono_tz doesn't have all tzdata symlink names",
    # Hypothesis finds overflow inputs on which the CPU streaming engine
    # disagrees with the in-memory result, so skip rather than xfail to avoid a
    # flaky XPASS/FAIL.
    "tests/unit/operations/test_statistics.py::test_skew_streaming_matches_in_memory[False]": "Compares the CPU streaming engine against the injected engine on random inputs",
    "tests/unit/operations/test_statistics.py::test_skew_streaming_matches_in_memory[True]": "Compares the CPU streaming engine against the injected engine on random inputs",
    "tests/unit/streaming/test_streaming_sort.py::test_sort_sets_sorted_flag[partitioned]": "Flaky in Polars itself; uses the CPU streaming engine and fails intermittently without the GPU engine",
    "tests/unit/io/test_lance.py::test_scan_lance_filter_pushdown_skips_files[streaming]": "Flaky upstream Polars cancellation race fixed by https://github.com/pola-rs/polars/pull/29774",
    "tests/unit/io/test_lance.py::test_scan_lance_filter_pushdown_skips_files[in-memory]": "Flaky upstream Polars cancellation race fixed by https://github.com/pola-rs/polars/pull/29774",
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
    "tests/unit/series/test_describe.py::test_series_describe_float": "https://github.com/NVIDIA/cudf/issues/19324",
    "tests/unit/series/test_describe.py::test_series_describe_int": "https://github.com/NVIDIA/cudf/issues/19324",
    "tests/unit/streaming/test_streaming.py::test_streaming_apply": "https://github.com/pola-rs/polars/issues/22558",
    "tests/unit/operations/test_group_by_dynamic.py::test_group_by_dynamic_agg_bad_input_types[str]": "TODO: Need to investigate why this fails in CI but passes locally. We should fallback to CPU for group_by_dynamic",
    "tests/unit/expr/test_exprs.py::test_exp_log1p[Float16-Float16]": "Flaky test: Small floating-point precision differences in exp/log1p results",
    # TODO: Investigate why these tests fail in CI but pass locally.
    "tests/unit/io/test_delta.py::test_scan_delta_extract_table_statistics_df": "schemas mismatch: dtypes different",
    "tests/unit/io/test_partition.py::test_sink_partitioned_no_columns_in_file_25535[scan_parquet-sink_parquet]": "Incorrect row count. Related to https://github.com/NVIDIA/cudf/issues/21428",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[0]": "ZeroDivisionError: division by zero",
    # Flaky deadlock test, may occur on rtxpro6000 only
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_in_mem_to_streaming_dispatch_deadlock_22641": "Flaky deadlock, may occur on rtxpro6000 only",
    # Short term adds in the aftermath of the rapidsmpf switch to get CI passing
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_local_with_async": "Flaky, otherwise TBD",
    "tests/unit/operations/test_join.py::test_join_where_nested_expr_21066": "Flaky, otherwise TBD",
    "tests/unit/io/test_scan.py::test_scan_metrics[True-parquet]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[True-csv]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[True-ndjson]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[True-ipc]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[False-parquet]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[False-csv]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[False-ndjson]": "Checks to IO metric logs specific to Polars CPU",
    "tests/unit/io/test_scan.py::test_scan_metrics[False-ipc]": "Checks to IO metric logs specific to Polars CPU",
    # Cloud credential provider tests that don't possible exercise cudf-polars.
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_serialization_custom_provider": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_gcp_skips_config_autoload": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_aws_import_error_with_requested_profile": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_aws_endpoint_url_scan_no_parameters": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_aws_endpoint_url_serde": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_aws_endpoint_url_with_storage_options": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_python_builder_cache": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_python_credentials_cache": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_no_pickle_option": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_aws_expiry": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_user_gcp_token_provider": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_auto_init_cache_key_memoize": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_cached_credential_provider_returns_copied_creds": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_cache_user_credential_provider": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_credential_provider_global_config": "Unused by cudf-polars",
    "tests/unit/io/cloud/test_credential_provider.py::test_cache_user_credential_provider_pickle": "Unused by cudf-polars",
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
            "tests/unit/sql/test_window_functions.py::test_window_frame_validation": 'sqlite3.OperationalError: near "GROUPS": syntax error',
            "tests/unit/sql/test_window_functions.py::test_window_total": 'sqlite3.OperationalError: near "NULLS": syntax error',
            "tests/unit/sql/test_window_functions.py::test_window_order_by_nulls_multiple_keys_29390[False-False]": 'sqlite3.OperationalError: near "NULLS": syntax error',
            "tests/unit/sql/test_window_functions.py::test_window_order_by_nulls_multiple_keys_29390[False-True]": 'sqlite3.OperationalError: near "NULLS": syntax error',
            "tests/unit/sql/test_window_functions.py::test_window_order_by_nulls_multiple_keys_29390[True-False]": 'sqlite3.OperationalError: near "NULLS": syntax error',
            "tests/unit/sql/test_window_functions.py::test_window_order_by_nulls_multiple_keys_29390[True-True]": 'sqlite3.OperationalError: near "NULLS": syntax error',
        }
    )


STREAMING_ENGINE_TESTS_TO_SKIP: Mapping[str, str] = {
    "tests/unit/operations/aggregation/test_aggregations.py::test_boolean_aggs": "float difference in std/var in the unit of least precision",
    # Crashes the worker instead of raising cleanly; skip rather than xfail
    # since a crashed worker never reports a result. See
    # https://github.com/NVIDIA/cudf/issues/24112
    "tests/unit/lazyframe/test_projections.py::test_projection_pushdown_union_len_pushdown_28657[concat]": "Materializing a huge virtual literal frame crashes the worker under the streaming engine",
    "tests/unit/lazyframe/test_projections.py::test_projection_pushdown_union_len_pushdown_28657[union]": "Materializing a huge virtual literal frame crashes the worker under the streaming engine",
    # No deterministic key sort (https://github.com/NVIDIA/cudf/issues/21641):
    # passes on some streaming runs and fails on others, so skip rather than
    # xfail to avoid a flaky XPASS/FAIL.
    "tests/unit/operations/test_group_by.py::test_group_by_unique_parametric[n_unique-True-True]": "non-deterministic key sort under the streaming engine",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_over_outer_join_tree[\\n    SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation\\n    WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey\\n      AND s_nationkey = n_nationkey AND n_name = 'FRANCE'\\n-=]": "Intermittently crashes when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_over_outer_join_tree[\\n    SELECT MAX(ps_supplycost) FROM partsupp, supplier, nation\\n    WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey\\n      AND s_nationkey = n_nationkey AND n_name = 'FRANCE'\\n-=]": "Intermittently crashes when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_over_outer_join_tree[\\n    SELECT 2 * MIN(ps_supplycost) FROM partsupp, supplier, nation\\n    WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey\\n      AND s_nationkey = n_nationkey AND n_name = 'FRANCE'\\n-<]": "Intermittently crashes when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_over_outer_join_tree[\\n            SELECT MIN(ps.ps_supplycost) FROM partsupp ps, supplier AS s, nation n\\n            WHERE n.n_name = 'FRANCE' AND s.s_nationkey = n.n_nationkey\\n              AND ps.ps_suppkey = s.s_suppkey AND ps.ps_partkey = part.p_partkey\\n            -=]": "Intermittently crashes when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_not_over_outer_join_tree[\\n    SELECT SUM(ps_supplycost) FROM partsupp, supplier, nation\\n    WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey\\n      AND s_nationkey = n_nationkey AND n_name = 'FRANCE'\\n-]": "Intermittently segfaults when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_not_over_outer_join_tree[\\n    SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation\\n    WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey\\n      AND s_nationkey = n_nationkey AND n_name = 'FRANCE'\\n AND ps_supplycost > 4-]": "Intermittently segfaults when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_not_over_outer_join_tree[\\n            SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation\\n            WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey\\n              AND s_nationkey = n_nationkey\\n            -]": "Intermittently segfaults when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_not_over_outer_join_tree[\\n    SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation\\n    WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey\\n      AND s_nationkey = n_nationkey AND n_name = 'FRANCE'\\n-AND p_partkey * 10 < s_suppkey]": "Intermittently segfaults when the process-global memory resource is swapped concurrently",
    "tests/unit/sql/test_correlated_subqueries.py::test_correlated_aggregate_not_over_outer_join_tree[\\n            SELECT MIN(ps2.ps_supplycost) FROM partsupp AS ps2\\n            WHERE ps2.ps_partkey = partsupp.ps_partkey\\n            -]": "Intermittently segfaults when the process-global memory resource is swapped concurrently",
    "tests/unit/operations/test_slice.py::test_slice_slice_pushdown": "Too slow for CI",
    "tests/unit/io/test_partition.py::test_partition_approximate_size": "Too slow for CI",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-parquet-async]": "Takes >60 seconds to run locally",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[glob-parquet-async]": "Takes >60 seconds to run locally",
    "tests/unit/lazyframe/test_optimizations.py::test_collapse_joins_combinations": "Too slow for CI",
    "tests/unit/operations/test_index_of.py::test_randomized": "Too slow for CI; marked as pytest.mark.slow",
    "tests/unit/streaming/test_streaming_sort.py::test_streaming_sort_varying_order_and_dtypes[sort_by0]": "Too slow for CI",
}

STREAMING_ENGINE_TESTS_TO_SKIP_SMALL_BLOCKSIZE: Mapping[str, str] = {
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
    "tests/unit/io/test_lazy_parquet.py::test_parquet_many_row_groups_12297": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_write_multiple_files": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_matches_buffered": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_parquet_defer_cached_reads[0]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_parquet_defer_cached_reads[1]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_file_posix_fadv[normal]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_read_roundtrip": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_file_posix_fadv[sequential]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_file_posix_fadv[willneed]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_file_posix_fadv[random]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_file_posix_fadv[foo_blah]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_projection[projection5]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_projection[projection4]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_projection[projection0]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_projection[projection6]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_multi_file": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_projection[projection2]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_eof_tail[4096]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_o_direct_projection[projection3]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_build_side.py::test_build_side_is_the_smaller_scan": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_build_side.py::test_build_side_survives_a_filter_on_the_large_side": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_transitive_keys_join_filtered_input_first": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_computed_group_key_does_not_borrow_a_column_distinct_count": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_plain_group_key_keeps_its_column_distinct_count": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_no_nulls_leaves_the_larger_dimension_last": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_null_count_drives_filter_selectivity": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_key_value_range_defers_a_fan_out_join[UInt64]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_key_value_range_defers_a_fan_out_join[Int64]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_key_value_range_defers_a_fan_out_join[Int32]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_key_value_range_defers_a_fan_out_join[UInt32]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_join_fused_predicate.py::test_fused_many_morsels_and_skew[streaming]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_join_fused_predicate.py::test_fused_many_morsels_and_skew[in-memory]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_join_fused_predicate.py::test_fused_rejected_batch_followed_by_accepted[streaming]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_join_fused_predicate.py::test_fused_rejected_batch_followed_by_accepted[in-memory]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_sampling_builds_smaller_left": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_skewed_and_large[auto]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_skewed_and_large[prefer_left]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_skewed_and_large[force_left]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_downstream_limit": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_skewed_and_large[force_right]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_skewed_and_large[prefer_right]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_ordered_and_is_in_keep_right_build": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/streaming/test_streaming_semi_join.py::test_sampling_hints": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_cse.py::test_cspe_predicate_overlap_drives_cache_removal[False-0]": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_cse.py::test_cspe_predicate_overlap_drives_cache_removal[True-8]": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_cse.py::test_cspe_cross_join_subplan_is_costed_after_pushdown": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_cse.py::test_cspe_shared_subplan_is_costed_with_the_common_bounds": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_two_keys_reading_one_column_count_it_once": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[String-519-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[Int32-519-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[Int32-519-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[String-519-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[dtype4-519-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[Boolean-519-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[dtype4-519-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[Boolean-519-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[dtype4-1056-False]": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[dtype4-1056-True]": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[dtype4-10432-False]": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[dtype4-10432-True]": "Hangs with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan[single-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan[single-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-parquet-async]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-parquet-sync]": "Too slow with --inject-gpu-engine-blocksize=small",
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
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Categorical-10432-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Categorical-10432-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-1056-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-1056-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-1056-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-1056-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-1056-False]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-1056-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_overflow_mean_partitioned_group_by_5194[Int32]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_overflow_mean_partitioned_group_by_5194[UInt32]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[True-scan_func0-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[True-scan_func1-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[True-<lambda>-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[False-scan_func0-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[False-scan_func1-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[False-scan_func2-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[False-<lambda>-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_hive.py::test_hive_partition_columns_contained_in_file[True-scan_func2-write_parquet]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_lance.py::test_scan_from_lance_slice_parametric": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/io/test_parquet.py::test_parametric_small_page_mask_filtering": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_getitem.py::test_lazyframe_getitem": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_self_join_keeps_each_copy_of_a_shared_column": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_repeated_dimension_filters_stay_on_their_own_relation": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_self_join_on_shared_column_names_is_reordered": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_correlated_multi_key_join_does_not_look_free": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_suffix_differing_between_joins_keeps_one_cluster": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_computed_projection_between_joins_is_not_dropped": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_with_columns_on_a_leaf_is_estimated": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_filter_above_a_rename_on_a_leaf": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_sort_leaf_is_estimated": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_union_leaf_is_estimated": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_rename_between_joins_keeps_one_cluster": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_sql_star_join_is_reordered": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_rename_of_a_join_key_between_joins": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_projection_between_joins_keeps_one_cluster": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_filter_on_a_leaf_is_estimated": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_unique_over_every_column_is_estimated": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_reordering_preserves_results_and_schema": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_unique_leaf_is_estimated": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_coalescing_reordering_preserves_results_and_schema": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_select_on_a_leaf_is_estimated": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_order.py::test_implied_key_equality_keeps_coalescing_sound": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_pushthrough.py::test_semi_join_above_a_selective_filter_stays": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_pushthrough.py::test_sql_left_join_is_null_idiom": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_pushthrough.py::test_semi_join_probing_no_more_rows_moves_below_inner_join[0.0]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_pushthrough.py::test_semi_join_probing_no_more_rows_moves_below_inner_join[10000.0]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/lazyframe/test_join_pushthrough.py::test_semi_join_moves_into_the_right_side_and_through_a_filter": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last[dtype4-100-True]": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_has_nulls.py::test_has_nulls_expr_parametric": "Too slow with --inject-gpu-engine-blocksize=small",
    "tests/unit/operations/test_join.py::test_cross_join_slice_pushdown": "Too slow with --inject-gpu-engine-blocksize=small",
}

STREAMING_ENGINE_EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/unit/sql/test_subqueries.py::test_scalar_subquery_multi_row_errors": "Correct polars.exceptions.ComputeError raised but it's in an ExceptionGroup",
    "tests/unit/functions/test_union.py::test_union_lazyframe_horizontal": "Correct polars.exceptions.ShapeError raised but it's in an ExceptionGroup",
    "tests/unit/functions/test_concat.py::test_concat_lazyframe_horizontal": "Correct polars.exceptions.ShapeError raised but it's in an ExceptionGroup",
    "tests/unit/functions/test_concat.py::test_concat_horizontal_zero_width_height_mismatch_26876": "https://github.com/NVIDIA/cudf/issues/21644",
    "tests/unit/operations/test_group_by.py::test_group_by_lit_series": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/aggregation/test_aggregations.py::test_item_too_many": "Correct polars.exceptions.ComputeError raised but it's in an ExceptionGroup",
    "tests/unit/operations/aggregation/test_aggregations.py::test_single_empty": "Correct polars.exceptions.ComputeError raised but it's in an ExceptionGroup",
    "tests/unit/operations/test_join.py::test_empty_outer_join_22206": "https://github.com/NVIDIA/cudf/issues/22084",
    "tests/unit/operations/test_replace.py::test_replace_invalid_old_dtype": "Correct InvalidOperationError raised but it's in an ExceptionGroup",
    "tests/unit/operations/test_window.py::test_over_literal_cum_sum_26800": "TODO: https://github.com/NVIDIA/cudf/pull/22048#discussion_r3238041970",
    "tests/unit/functions/test_concat.py::test_concat_horizontal_lazy_strict_raises_shape_error_27415": "horizontal-concat strict height-mismatch raised inside an ExceptionGroup under the streaming engine",
    "tests/unit/io/test_io_plugin.py::test_defer_validate_true": "correct SchemaError raised but wrapped in an ExceptionGroup under the streaming engine",
    "tests/unit/lazyframe/test_projections.py::test_merge_sorted_projection_pd": "https://github.com/NVIDIA/cudf/issues/23055",
    "tests/unit/operations/test_slice.py::test_hconcat_tail_unequal_heights_strict_raises_27552": "horizontal-concat strict height-mismatch raised inside an ExceptionGroup under the streaming engine",
}

STREAMING_ENGINE_EXPECTED_FAILURES_SMALL_BLOCKSIZE: Mapping[str, str] = {
    "tests/unit/operations/test_slice.py::test_slice_pushdown_literal_projection_14349": "https://github.com/NVIDIA/cudf/issues/22072",
    "tests/unit/functions/range/test_linear_space.py::test_linear_space_num_samples_expr": "https://github.com/NVIDIA/cudf/issues/22072",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[1]": "https://github.com/NVIDIA/cudf/issues/22075",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[4]": "https://github.com/NVIDIA/cudf/issues/22075",
}


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark known failing tests."""
    num_shards = config.getoption("cudf_polars_num_shards")
    if num_shards > 1:
        shard_id = config.getoption("cudf_polars_shard_id")
        items[:], deselected = partition_items_by_shard(items, shard_id, num_shards)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
    if config.getoption("--inject-gpu-engine-raise-on-fail"):
        # Don't xfail tests if running without fallback
        return
    with_streaming_engine = config.getoption("--inject-gpu-engine") == "spmd"
    with_small_blocksize = (
        with_streaming_engine
        and config.getoption("--inject-gpu-engine-blocksize") == "small"
    )
    skips = dict(TESTS_TO_SKIP)
    xfails = dict(EXPECTED_FAILURES)
    if with_streaming_engine:
        skips.update(STREAMING_ENGINE_TESTS_TO_SKIP)
        xfails.update(STREAMING_ENGINE_EXPECTED_FAILURES)
        if with_small_blocksize:
            skips.update(STREAMING_ENGINE_TESTS_TO_SKIP_SMALL_BLOCKSIZE)
            xfails.update(STREAMING_ENGINE_EXPECTED_FAILURES_SMALL_BLOCKSIZE)
    for item in items:
        if (skip_reason := skips.get(item.nodeid)) is not None:
            item.add_marker(pytest.mark.skip(reason=skip_reason))
        elif (xfail_reason := xfails.get(item.nodeid)) is not None:
            item.add_marker(pytest.mark.xfail(reason=xfail_reason))
