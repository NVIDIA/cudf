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

examples_timeout=30m
kafka_timeout=30m
if [[ ${RAPIDS_BUILD_TYPE:-} == pull-request ]]; then
    examples_timeout=5m
    kafka_timeout=2m
fi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests from the libcudf-tests and libcudf-streaming-tests packages
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

SUITEERROR=0

# TEMP: WSL diagnostics
nproc; free -g; cat /proc/meminfo | head -5
(while true; do nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,pstate,clocks.sm,temperature.gpu,power.draw --format=csv,noheader; free -m | sed -n 2p; cat /proc/loadavg; sleep 30; done) > "${RAPIDS_TESTS_DIR:-/tmp}/../wsl-monitor.log" 2>&1 &
WSL_MON_PID=$!

if [[ "${RUN_LIBCUDF_TESTS}" == "true" ]]; then
    rapids-logger "TEMP: driver probe"
    python ci/wsl_driver_probe.py
    gtest_dir="${CONDA_PREFIX}/bin/gtests/libcudf"
    rapids-logger "TEMP: GROUPBY_TEST per-test timing"
    timeout 10m "${gtest_dir}/GROUPBY_TEST" 2>&1 | grep -E '^\[ +OK +\]' | sed -E 's/.*\] (.*) \(([0-9]+) ms\)/\2 \1/' | sort -rn | head -25
    rapids-logger "TEMP: ORC_TEST with 5m timeout"
    timeout 5m "${gtest_dir}/ORC_TEST" 2>&1 | grep -E '^\[ +(OK|RUN|FAILED) +\]' | tail -15
    echo "TEMP: ORC_TEST exit $?"
    kill "${WSL_MON_PID}" || true
    cat "${RAPIDS_TESTS_DIR:-/tmp}/../wsl-monitor.log" | tail -40
    exit 1
    rapids-logger "Run libcudf gtests"
    timeout 30m ./ci/run_cudf_ctests.sh -j20
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_EXAMPLES_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf examples"
    timeout "${examples_timeout}" ./ci/run_cudf_examples.sh
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_LIBCUDF_KAFKA_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf_kafka gtests"
    timeout "${kafka_timeout}" ./ci/run_cudf_kafka_ctests.sh -j20
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

kill "${WSL_MON_PID}" || true
cat "${RAPIDS_TESTS_DIR:-/tmp}/../wsl-monitor.log"
rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
