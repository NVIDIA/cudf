#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare coverage from the local cudf-polars suite and the checked-out upstream
# Polars suite. Run ci/test_cudf_polars_polars_tests.sh first to prepare polars/.
set -euo pipefail

repo_root="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
output_dir=$(realpath -m "${1:-${repo_root}/coverage-results/cudf-polars-comparison}")
config="${repo_root}/ci/cudf_polars_coverage_analysis.toml"
polars_dir="${POLARS_DIR:-${repo_root}/polars}"

if [[ ! -d "${polars_dir}/py-polars/tests" ]]; then
    echo "Missing ${polars_dir}/py-polars/tests. Run ci/test_cudf_polars_polars_tests.sh first." >&2
    exit 2
fi

polars_version=$(python -c 'import polars; print(polars.__version__)')
polars_sha=$(git -C "${polars_dir}" rev-parse HEAD)
if [[ "${polars_sha}" != "$(git -C "${polars_dir}" rev-parse --verify "refs/tags/py-${polars_version}^{commit}")" ]]; then
    echo "Expected the Polars checkout at ${polars_dir} to be py-${polars_version}, got ${polars_sha}." >&2
    exit 2
fi
cudf_sha=$(git -C "${repo_root}" rev-parse HEAD)
pytest_args=(
    --numprocesses=4 --dist=worksteal
    --cov=cudf_polars --cov-context=test --cov-config="${config}"
    --cov-report= --maxfail=0
)

mkdir -p "${output_dir}"
rm -f "${output_dir}/.coverage.local" \
    "${output_dir}/.coverage.upstream-in-memory" \
    "${output_dir}/.coverage.upstream-spmd" \
    "${output_dir}/coverage-comparison.json"

local_status=0
COVERAGE_FILE="${output_dir}/.coverage.local" \
    "${repo_root}/ci/run_cudf_polars_pytests.sh" \
    "${pytest_args[@]}" || local_status=$?

# Keep engine data separate so later analysis can distinguish configuration
# gaps from missing upstream assertions, even when test node IDs are identical.
export POLARS_DIR="${polars_dir}" CUDF_POLARS_GPU_RUN_LIST=""
upstream_in_memory_status=0
COVERAGE_FILE="${output_dir}/.coverage.upstream-in-memory" \
    "${repo_root}/ci/run_cudf_polars_polars_tests.sh" --engine in-memory \
    "${pytest_args[@]}" || upstream_in_memory_status=$?
upstream_spmd_status=0
COVERAGE_FILE="${output_dir}/.coverage.upstream-spmd" \
    "${repo_root}/ci/run_cudf_polars_polars_tests.sh" --engine spmd \
    --inject-gpu-engine-blocksize default \
    "${pytest_args[@]}" || upstream_spmd_status=$?

python "${repo_root}/ci/analyze_cudf_polars_coverage.py" \
    --local-data "${output_dir}/.coverage.local" \
    --upstream-data "${output_dir}/.coverage.upstream-in-memory" \
    --upstream-data "${output_dir}/.coverage.upstream-spmd" \
    --polars-version "${polars_version}" \
    --polars-sha "${polars_sha}" \
    --cudf-sha "${cudf_sha}" \
    --local-exit-status "${local_status}" \
    --upstream-in-memory-exit-status "${upstream_in_memory_status}" \
    --upstream-spmd-exit-status "${upstream_spmd_status}" \
    --output-dir "${output_dir}"

echo "Wrote coverage data and line-level comparison to ${output_dir}"
if ((local_status || upstream_in_memory_status || upstream_spmd_status)); then
    echo "One or more test suites failed; see suite_exit_statuses in coverage-comparison.json." >&2
    exit 1
fi
