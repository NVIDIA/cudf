#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

# Support invoking run_cudf_polars_pytests.sh outside the script directory
# Assumption, polars has been cloned in the root of the repo.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../polars/

DESELECTED_TESTS=(
    "tests/unit/meta/test_polars_import.py::test_polars_import" # relies on a polars built in place
    "tests/unit/streaming/test_streaming_sort.py::test_streaming_sort[True]" # relies on polars built in debug mode
    "tests/docs/test_user_guide.py" # No dot binary in CI image
    "tests/unit/operations/test_join.py::test_join_4_columns_with_validity" # fails in some systems, see https://github.com/pola-rs/polars/issues/19870
    "tests/unit/io/test_csv.py::test_read_web_file" # fails in rockylinux8 due to SSL CA issues
    # TODO: Debug and re-enable the following tests
    "tests/unit/sql/test_distinct.py::test_distinct_with_full_outer_join" # SQLite in CI doesn't support FULL OUTER JOIN
    "tests/unit/sql/test_distinct.py::test_distinct_basic_single_column"
    "tests/unit/sql/test_distinct.py::test_distinct_basic_multiple_columns"
    "tests/unit/sql/test_distinct.py::test_distinct_basic_all_columns"
    "tests/unit/sql/test_distinct.py::test_distinct_with_left_join_nulls"
    "tests/unit/sql/test_distinct.py::test_distinct_with_nulls_handling"
    "tests/unit/sql/test_window_functions.py::test_window_function_with_nulls"
    "tests/unit/io/test_sink.py::test_mkdir[in-memory-scan_parquet-sink_parquet]" # kvikio file creation error in CI
    "tests/unit/io/test_sink.py::test_mkdir[in-memory-scan_csv-sink_csv]" # kvikio file creation error in CI
    "tests/unit/io/test_sink.py::test_mkdir[in-memory-scan_ndjson-sink_ndjson]" # kvikio file creation error in CI
    "tests/unit/io/test_sink.py::test_mkdir[streaming-scan_parquet-sink_parquet]" # kvikio file creation error in CI
    "tests/unit/io/test_sink.py::test_mkdir[streaming-scan_csv-sink_csv]" # kvikio file creation error in CI
    "tests/unit/io/test_sink.py::test_mkdir[streaming-scan_ndjson-sink_ndjson]" # kvikio file creation error in CI
    "tests/unit/io/test_write.py::test_write_async[read_parquet-<lambda>]" # kvikio file creation error in CI
    "tests/unit/io/test_write.py::test_write_async[<lambda>-<lambda>0]" # kvikio file creation error in CI
    "tests/unit/io/test_write.py::test_write_async[<lambda>-<lambda>2]" # kvikio file creation error in CI
    "tests/unit/operations/test_random.py::test_shuffle_group_by_reseed" # https://github.com/NVIDIA/cudf/issues/22964
)

if [[ $(arch) == "aarch64" ]]; then
    # The binary used for TPC-H generation is compiled for x86_64, not aarch64.
    DESELECTED_TESTS+=("tests/benchmark/test_pdsh.py::test_pdsh")
    # The connectorx package is not available on arm
    DESELECTED_TESTS+=("tests/unit/io/database/test_read.py::test_read_database")
    # The necessary timezone information cannot be found in our CI image.
    DESELECTED_TESTS+=("tests/unit/io/test_parquet.py::test_parametric_small_page_mask_filtering")
    DESELECTED_TESTS+=("tests/unit/testing/test_assert_series_equal.py::test_assert_series_equal_parametric")
    DESELECTED_TESTS+=("tests/unit/operations/test_join.py::test_join_4_columns_with_validity")
else
    # Ensure that we don't run dbgen when it uses newer symbols than supported by the glibc version in the CI image.
    # Allow errors since any of these commands could produce empty results that would cause the script to fail.
    set +e
    glibc_minor_version=$(ldd --version | head -1 | grep -o "[0-9]\.[0-9]\+" | tail -1 | cut -d '.' -f2)
    latest_glibc_symbol_found=$(nm py-polars/tests/benchmark/data/pdsh/dbgen/dbgen | grep GLIBC | grep -o "[0-9]\.[0-9]\+" | sort --version-sort | tail -1 | cut -d "." -f 2)
    set -e
    if [[ ${glibc_minor_version} -lt ${latest_glibc_symbol_found} ]]; then
        DESELECTED_TESTS+=("tests/benchmark/test_pdsh.py::test_pdsh")
    fi
fi

DESELECTED_TESTS_STR=$(printf -- " --deselect %s" "${DESELECTED_TESTS[@]}")

# Don't quote the `DESELECTED_...` variable because `pytest` can't handle
# multiple quoted arguments inline
# shellcheck disable=SC2086
# Fail fast (-x) rather than trying to continue because failed tests pollute the state
echo "Run polars tests with injected in-memory GPU engine"
python "${TIMEOUT_TOOL_PATH}" --enable-python 5400 \
   python -m pytest \
       --import-mode=importlib \
       --cache-clear \
       -x \
       -m "" \
       -p cudf_polars.testing.inject_gpu_engine \
       -n 4 \
       --dist=worksteal \
       --tb=native \
       --durations 10 --durations-min 10 \
       $DESELECTED_TESTS_STR \
       "$@" \
       py-polars/tests \
       --inject-gpu-engine in-memory

# TODO(ResourceWarning): https://github.com/NVIDIA/cudf/issues/22181
echo "Run polars tests with injected SPMD GPU engine, small blocksize"
CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE=805306368 \
CUDF_POLARS__EXECUTOR__FALLBACK_MODE=silent \
python "${TIMEOUT_TOOL_PATH}" --enable-python 5400 \
   python -m pytest \
       --import-mode=importlib \
       --cache-clear \
       -x \
       -m "" \
       -p cudf_polars.testing.inject_gpu_engine \
       -W ignore::ResourceWarning \
       -n 4 \
       --dist=worksteal \
       --tb=native \
       --durations 10 --durations-min 10 \
       $DESELECTED_TESTS_STR \
       "$@" \
       py-polars/tests \
       --inject-gpu-engine spmd \
       --inject-gpu-engine-blocksize small
