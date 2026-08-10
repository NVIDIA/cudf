#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_other

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest dask_cudf"
timeout 30m ./ci/run_dask_cudf_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=../.coveragerc \
  --cov=dask_cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/dask-cudf-coverage.xml" \
  --cov-report=term

rapids-logger "pytest cudf_kafka"
timeout 30m ./ci/run_cudf_kafka_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-kafka.xml"

rapids-logger "pytest custreamz"
timeout 30m ./ci/run_custreamz_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-custreamz.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=../.coveragerc \
  --cov=custreamz \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/custreamz-coverage.xml" \
  --cov-report=term

rapids-logger "pytest cudf-polars"
# Avoid oversubscribing the CPU: NJOBS pytest-xdist workers each get nproc/NJOBS Polars threads
NJOBS=4
NPROC=$(nproc)
export POLARS_MAX_THREADS=$(( NPROC / NJOBS > 0 ? NPROC / NJOBS : 1 ))

export LIBCUDF_NUM_HOST_WORKERS=2
export OMP_NUM_THREADS=1
export RAY_core_worker_num_server_call_thread=2
export RAY_num_server_call_thread=2
export RAY_prestart_worker_first_driver=0
export RAY_worker_num_grpc_internal_threads=2

echo "n-jobs=${NJOBS}, n-proc=${NPROC}, polars-max-threads=${POLARS_MAX_THREADS}"

# Fail fast (-x) rather than trying to continue because failed tests pollute the state
./ci/run_cudf_polars_pytests.sh \
  -x \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars.xml" \
  --numprocesses=${NJOBS} \
  --dist=loadgroup \
  --cov-config=./pyproject.toml \
  --cov=cudf_polars \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-polars-coverage.xml" \
  --cov-report=term \
  --durations=10 --durations-min=10 \
  -ra

rapids-logger "pytest cudf_streaming"
timeout 30m ./ci/run_cudf_streaming_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-streaming.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
