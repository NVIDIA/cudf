#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_REPOSITORY=NVIDIA/cudf
export RAPIDS_BASE_BRANCH=main
export RAPIDS_REF_NAME=pull-request/25532
export RAPIDS_SHA=4b383a45d42d6ce75dc96fb9c50be380c624a9a3
export CUDF_POLARS_DIAGNOSTICS_DIR="$PWD/artifacts/arm64-polars"
export RAPIDS_TESTS_DIR="$CUDF_POLARS_DIAGNOSTICS_DIR/test-results"
export RAPIDS_COVERAGE_DIR="$CUDF_POLARS_DIAGNOSTICS_DIR/coverage"
{ read -r _; IFS=, read -r CUDF_POLARS_GPU_UUID _; } < "$CUDF_POLARS_DIAGNOSTICS_DIR/gpus.csv"
export CUDF_POLARS_GPU_UUID

export PYTHONPATH="$PWD/ci/arm64_polars_diagnostics${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_PLUGINS=repro_diagnostics
if [[ "${CUDF_POLARS_DIAGNOSTICS_TRIAL}" == warmup-* ]]; then
    # Coverage gates do not apply to the selected worker history.
    export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --no-cov -W ignore::pytest_cov.CovDisabledWarning"
fi

# The original run's packages were rebuilt at the same source revision on retry.
rapids-download-from-github() {
    GITHUB_RUN_ID=45518453978 command rapids-download-from-github "$@"
}
export -f rapids-download-from-github

# The frozen checkout has stale test calls to the ColRef constructor.
git -C replay-source apply --unidiff-zero "$PWD/ci/arm64_polars_diagnostics/colref-tests.patch"
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck ]]; then
    cp ci/arm64_polars_diagnostics/memcheck-pytest.sh replay-source/ci/run_cudf_polars_pytests.sh
elif [[ "$CUDF_POLARS_REPEAT_REPLAY" == true ]]; then
    # Resolve the exact packages once and run fresh pytest processes on the same GPU.
    mv replay-source/ci/run_cudf_polars_pytests{,-once}.sh
    cp ci/arm64_polars_diagnostics/repeat-pytest.sh replay-source/ci/run_cudf_polars_pytests.sh
    chmod +x replay-source/ci/run_cudf_polars_pytests.sh
fi
cd replay-source
env POLARS_VERSIONS=latest bash ci/test_python_cudf_polars.sh
