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
    timeout 5m ./ci/run_cudf_streaming_ctests.sh -j1
    SUITEERROR=$?
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
