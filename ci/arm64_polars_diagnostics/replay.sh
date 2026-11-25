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

# Both historical shuffle faults occurred on this GPU. Keep a comparison on
# the normal pool and skip the full test suite on other allocations.
target_gpu=GPU-ffa4d437-0552-ea7a-0e5d-0fe4bf4fd918
gpu_inventory=$(< "$CUDF_POLARS_DIAGNOSTICS_DIR/gpus.csv")
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == gdb-[0-9][0-9] ]] &&
    [[ "$gpu_inventory" != *"${target_gpu},"* ]]; then
    echo 'Target GPU was not assigned' | tee "$CUDF_POLARS_DIAGNOSTICS_DIR/skipped.txt"
    exit 0
fi
nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER,RESET_STATUS > "$CUDF_POLARS_DIAGNOSTICS_DIR/gpu-health-before.txt" 2>&1 || true

export PYTHONPATH="$PWD/ci/arm64_polars_diagnostics${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_PLUGINS=repro_diagnostics
if [[ "${CUDF_POLARS_DIAGNOSTICS_TRIAL}" == warmup-* ]]; then
    # Coverage gates do not apply to the selected worker history.
    export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --no-cov -W ignore::pytest_cov.CovDisabledWarning"
fi

# The replay uses the package artifacts from the failing run even when this workflow is rerun.
rapids-download-from-github() {
    GITHUB_RUN_ID=45449689051 command rapids-download-from-github "$@"
}
export -f rapids-download-from-github

# The frozen checkout has stale test calls to the ColRef constructor.
git -C replay-source apply --unidiff-zero "$PWD/ci/arm64_polars_diagnostics/colref-tests.patch"
if [[ "$gpu_inventory" == *"${target_gpu},"* ]]; then
    # Resolve the exact packages once and run fresh pytest processes on the same GPU.
    mv replay-source/ci/run_cudf_polars_pytests{,-once}.sh
    cp ci/arm64_polars_diagnostics/repeat-pytest.sh replay-source/ci/run_cudf_polars_pytests.sh
    chmod +x replay-source/ci/run_cudf_polars_pytests.sh
fi
cd replay-source
env POLARS_VERSIONS=latest bash ci/test_python_cudf_polars.sh
