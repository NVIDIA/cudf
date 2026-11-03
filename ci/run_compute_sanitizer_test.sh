#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

# This script runs compute-sanitizer on a single libcudf test executable
# Usage: ./run_compute_sanitizer_test.sh TOOL_NAME TEST_NAME [additional gtest args...]
# Example: ./run_compute_sanitizer_test.sh memcheck AST_TEST
# Example: ./run_compute_sanitizer_test.sh racecheck COMPRESSION_TEST --gtest_filter=CompressionTest.*

if [ $# -lt 2 ]; then
  echo "Error: Tool and test name required"
  echo "Usage: $0 TOOL_NAME TEST_NAME [additional gtest args...]"
  echo "  TOOL_NAME: compute-sanitizer tool (memcheck, racecheck, initcheck, synccheck)"
  echo "  TEST_NAME: libcudf test name"
  exit 1
fi

TOOL_NAME="${1}"
shift
TEST_NAME="${1}"
shift

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-logger "Create test environment"
rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Running compute-sanitizer --tool ${TOOL_NAME} on ${TEST_NAME}"

# Set environment variables as per ci/run_cudf_memcheck_ctests.sh
export GTEST_CUDF_RMM_MODE=cuda
# Allows tests to know they are in a compute-sanitizer run
export LIBCUDF_${TOOL_NAME^^}_ENABLED=1

# Navigate to test installation directory
TEST_DIR="${CONDA_PREFIX}/bin/gtests/libcudf"
TEST_EXECUTABLE="${TEST_DIR}/${TEST_NAME}"

if [ ! -x "${TEST_EXECUTABLE}" ]; then
  rapids-logger "Error: Test executable ${TEST_EXECUTABLE} not found or not executable"
  exit 1
fi

CS_EXCLUDE_NAMES="kns=nvcomp,kns=zstd,kns=_no_sanitize,kns=_no_${TOOL_NAME}"
CS_STREAM_ORDERED_RACES="all"
if [ "${TEST_NAME}" == "LARGE_STRINGS_TEST" ]; then
  CS_STREAM_ORDERED_RACES="no"
fi

# Run compute-sanitizer on the specified test
compute-sanitizer \
  --tool "${TOOL_NAME}" \
  --force-blocking-launches \
  --kernel-name-exclude "${CS_EXCLUDE_NAMES}" \
  --track-stream-ordered-races "${CS_STREAM_ORDERED_RACES}" \
  --error-exitcode=1 \
  "${TEST_EXECUTABLE}" \
  "$@"

EXITCODE=$?

# Clean up environment variables
unset GTEST_CUDF_RMM_MODE
unset LIBCUDF_${TOOL_NAME^^}_ENABLED

rapids-logger "compute-sanitizer --tool ${TOOL_NAME} on ${TEST_NAME} exiting with value: $EXITCODE"
exit $EXITCODE
