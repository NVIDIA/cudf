#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_python_common.sh test_python_other

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
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-polars-coverage.xml" \
  --cov-report=term \
  --durations=50 --durations-min=1
