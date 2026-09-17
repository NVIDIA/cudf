#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_cpp_common.sh

RUN_LIBCUDF_TESTS="${RUN_LIBCUDF_TESTS:-true}"
RUN_EXAMPLES_TESTS="${RUN_EXAMPLES_TESTS:-true}"
RUN_LIBCUDF_KAFKA_TESTS="${RUN_LIBCUDF_KAFKA_TESTS:-true}"
RUN_LIBCUDF_STREAMING_TESTS="${RUN_LIBCUDF_STREAMING_TESTS:-true}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests from the libcudf-tests and libcudf-streaming-tests packages
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

SUITEERROR=0

if [[ "${RUN_LIBCUDF_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf gtests"
    timeout 30m ./ci/run_cudf_ctests.sh -j20
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_LIBCUDF_JIT_CACHE_DIAGNOSTICS:-false}" == "true" ]]; then
    rapids-logger "Run cache-disabled JIT cast diagnostics"
    test_dir="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf"
    test_args=(--gtest_filter="JITExpressionTest.Cast:JITExpressionTest.DecimalCast")

    # Profile all AST tests once, on the representative CUDA 13 amd64 matrix entry.
    if [[ "${RUN_LIBCUDF_JIT_AST_PROFILE:-false}" == "true" &&
          "${RAPIDS_CUDA_VERSION}" == "13.3.0" && "$(uname -m)" == "x86_64" ]]; then
        rapids-logger "Profile cache-disabled AST tests"
        test_args=(--gtest_print_time=1)
    fi

    timeout 5m "${test_dir}/AST_TEST" \
        "${test_args[@]}"
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_EXAMPLES_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf examples"
    timeout 30m ./ci/run_cudf_examples.sh
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_LIBCUDF_KAFKA_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf_kafka gtests"
    timeout 30m ./ci/run_cudf_kafka_ctests.sh -j20
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_LIBCUDF_STREAMING_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf_streaming gtests"
    # cudf_streaming contains distributed tests, and running tests in
    # parallel results in resource starvation CI env.
    # Multiple MPI ranks execute the same GTest binary concurrently. GTest's
    # directory-mode XML output chooses file names without interprocess
    # synchronization, so ranks can corrupt a shared report. Keep XML reports
    # enabled for the non-distributed suites above, but disable them here.
    saved_gtest_output="${GTEST_OUTPUT}"
    unset GTEST_OUTPUT
    timeout 5m ./ci/run_cudf_streaming_ctests.sh -j1
    SUITEERROR=$?
    export GTEST_OUTPUT="${saved_gtest_output}"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
