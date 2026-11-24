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

# Run the large-memory tests (skipped by default) in nightlies; they skip themselves on small GPUs
if [[ "${RAPIDS_BUILD_TYPE:-}" == "nightly" ]]; then
    export GTEST_CUDF_LARGE_MEMORY_TESTS=1
fi

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

if [[ ${RAPIDS_BUILD_TYPE:-} == pull-request && ${RUN_LIBCUDF_TESTS} == true ]]; then
    if ! command -v zstd >/dev/null; then
        rapids-logger "Skipping RTCX cache compression benchmark: zstd is unavailable"
    elif [[ -z $(find "${LIBCUDF_KERNEL_CACHE_PATH}" -type f -print -quit) ]]; then
        rapids-logger "Skipping RTCX cache compression benchmark: cache is empty"
    else
        # Match the cache action's zstd settings and stream the archive so the cache stays usable.
        for level in 3 10 19; do
            started=${SECONDS}
            if bytes=$(tar --posix --sort=name --pax-option=delete=atime,delete=ctime \
                -cf - -C "$(dirname "${LIBCUDF_KERNEL_CACHE_PATH}")" "$(basename "${LIBCUDF_KERNEL_CACHE_PATH}")" |
                timeout 5m zstd -q -c -T0 --long=30 "-${level}" | wc -c); then
                rapids-logger "RTCX cache zstd level ${level}: ${bytes} bytes in $((SECONDS - started)) s"
            else
                rapids-logger "RTCX cache compression benchmark failed at zstd level ${level}"
            fi
        done
    fi
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
