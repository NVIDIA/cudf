#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUN_PYLIBCUDF_TESTS="${RUN_PYLIBCUDF_TESTS:-true}"
RUN_CUDF_TESTS="${RUN_CUDF_TESTS:-true}"
RUN_CUDF_BENCHMARK_TESTS="${RUN_CUDF_BENCHMARK_TESTS:-true}"

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_pylibcudf test_python_cudf

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

if [[ "${RUN_PYLIBCUDF_TESTS}" == "true" ]]; then
  rapids-logger "pytest pylibcudf"
  timeout 40m ./ci/run_pylibcudf_pytests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibcudf.xml" \
    --numprocesses=8 \
    --dist=worksteal \
    --cov-config=.coveragerc \
    --cov=pylibcudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibcudf-coverage.xml" \
    --cov-report=term \
    --durations=50 --durations-min=1
fi

version_gte() {
  [ "$2" = "$(echo -e "$2\n$1" | sort -V | head -n1)" ]
}

if [[ "${RUN_CUDF_TESTS}" == "true" ]]; then
  rapids-logger "pytest cudf"
  timeout 40m ./ci/run_cudf_pytests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
    --numprocesses=8 \
    --dist=worksteal \
    --cov-config=../.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-coverage.xml" \
    --cov-report=term \
    --durations=50 --durations-min=1
fi

# Run benchmarks with both cudf and pandas to ensure compatibility is maintained.
# Benchmarks are run in DEBUG_ONLY mode, meaning that only small data sizes are used.
# Therefore, these runs only verify that benchmarks are valid.
# They do not generate meaningful performance measurements.

if [[ "${RUN_CUDF_BENCHMARK_TESTS}" == "true" ]]; then
  rapids-logger "pytest for cudf benchmarks"
  timeout 40m ./ci/run_cudf_pytest_benchmarks.sh \
    --benchmark-disable \
    --numprocesses=8 \
    --dist=worksteal \
    --cov-config=.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-coverage.xml" \
    --cov-report=term \
    --durations=50 --durations-min=1

  rapids-logger "pytest for cudf benchmarks using pandas"
  timeout 40m ./ci/run_cudf_pandas_pytest_benchmarks.sh \
    --benchmark-disable \
    --numprocesses=8 \
    --dist=worksteal \
    --cov-config=.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-pandas-coverage.xml" \
    --cov-report=term \
    --durations=50 --durations-min=1
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
