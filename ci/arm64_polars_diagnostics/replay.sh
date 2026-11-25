#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_REPOSITORY=NVIDIA/cudf
export RAPIDS_BASE_BRANCH=main
export RAPIDS_REF_NAME=pull-request/25452
export RAPIDS_SHA=ed03e588cefe03804174c2141f4893d2f4f01833
export CUDF_POLARS_DIAGNOSTICS_DIR="$PWD/artifacts/arm64-polars"
export RAPIDS_TESTS_DIR="$CUDF_POLARS_DIAGNOSTICS_DIR/test-results"
export RAPIDS_COVERAGE_DIR="$CUDF_POLARS_DIAGNOSTICS_DIR/coverage"
mkdir -p "$CUDF_POLARS_DIAGNOSTICS_DIR"
nvidia-smi --query-gpu=uuid,driver_version,pci.bus_id --format=csv > "$CUDF_POLARS_DIAGNOSTICS_DIR/gpus.csv"
export PYTHONPATH="$PWD/ci/arm64_polars_diagnostics${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_PLUGINS=repro_diagnostics
if [[ "${CUDF_POLARS_DIAGNOSTICS_TRIAL}" == stress-* ]]; then
    # The stress lanes run selected cases rather than the coverage suite.
    export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --no-cov -W ignore::pytest_cov.CovDisabledWarning"
fi

# The replay uses the package artifacts from the failing run even when this workflow is rerun.
rapids-download-from-github() {
    GITHUB_RUN_ID=45449689051 command rapids-download-from-github "$@"
}
export -f rapids-download-from-github

# The frozen checkout has stale test calls to the ColRef constructor.
git -C replay-source apply --unidiff-zero "$PWD/ci/arm64_polars_diagnostics/colref-tests.patch"
cd replay-source
env POLARS_VERSIONS=latest bash ci/test_python_cudf_polars.sh
