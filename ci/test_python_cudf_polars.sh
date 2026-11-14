#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Wheel compatibility tests cover the earliest supported Polars release. Select the
# newest release here so the two package formats cover opposite endpoints.
read -r -a POLARS_COMPAT_VERSIONS <<< "$(python ci/utils/get_matrix_values.py dependencies.yaml test_cudf_polars_compat polars_compat_version)"
case "${POLARS_VERSIONS:-latest}" in
  earliest) POLARS_COMPAT_VERSION="${POLARS_COMPAT_VERSIONS[0]}" ;;
  latest) POLARS_COMPAT_VERSION="${POLARS_COMPAT_VERSIONS[-1]}" ;;
  *)
    echo "Unsupported POLARS_VERSIONS=${POLARS_VERSIONS}" >&2
    exit 1
    ;;
esac
export CUDF_EXTRA_DEPENDENCY_MATRIX="polars_compat_version=${POLARS_COMPAT_VERSION}"

# Channel with the cudf-polars-quent package built in the previous step. The package is
# abi3 with no CUDA dependency, so it is requested as a stable artifact without --cuda.
CUDF_POLARS_QUENT_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf-polars-quent cudf --stable)")
export CUDF_POLARS_QUENT_CHANNEL

source ./ci/test_python_common.sh test_python_other test_cudf_polars_compat test_cudf_polars_quent

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env

rapids-logger "pytest cudf-polars"
# Fail fast (-x) rather than trying to continue because failed tests pollute the state.
./ci/run_cudf_polars_pytests.sh \
  -x \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars.xml" \
  --numprocesses=4 \
  --dist=worksteal \
  --cov-config=./pyproject.toml \
  --cov=cudf_polars \
  --cov-fail-under=100 \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-polars-coverage.xml" \
  --cov-report=term-missing:skip-covered \
  --durations=50 --durations-min=1
