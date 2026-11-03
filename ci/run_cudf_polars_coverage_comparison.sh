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
if [[ "${polars_version}" != 2.0.* && "${polars_version}" != 2.1.* ]]; then
    echo "Expected an upstream-supported Polars 2.0 or 2.1 installation, got ${polars_version}." >&2
    exit 2
fi
polars_sha=$(git -C "${polars_dir}" rev-parse HEAD)
polars_tag=$(git -C "${polars_dir}" describe --tags --exact-match HEAD 2>/dev/null || true)
if [[ "${polars_tag}" != "py-${polars_version}" ]]; then
    echo "Expected the Polars checkout at ${polars_dir} to be py-${polars_version}, got ${polars_tag:-an untagged revision}." >&2
    exit 2
fi
cudf_sha=$(git -C "${repo_root}" rev-parse HEAD)

mkdir -p "${output_dir}"
rm -f "${output_dir}/.coverage.local" "${output_dir}/.coverage.upstream"

local_status=0
COVERAGE_FILE="${output_dir}/.coverage.local" \
    "${repo_root}/ci/run_cudf_polars_pytests.sh" \
    --numprocesses=4 --dist=worksteal \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" --cov-report= || local_status=$?

# pytest-cov merges xdist data itself. --cov-append keeps both upstream engine
# runs in one file without mixing their data with the local suite. Continue
# after test failures so both configurations contribute diagnostic coverage.
upstream_in_memory_status=0
POLARS_DIR="${polars_dir}" COVERAGE_FILE="${output_dir}/.coverage.upstream" \
    "${repo_root}/ci/run_cudf_polars_polars_tests.sh" --engine in-memory \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" \
    --cov-report= --cov-append --maxfail=0 || upstream_in_memory_status=$?
upstream_spmd_status=0
POLARS_DIR="${polars_dir}" COVERAGE_FILE="${output_dir}/.coverage.upstream" \
    "${repo_root}/ci/run_cudf_polars_polars_tests.sh" --engine spmd \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" \
    --cov-report= --cov-append --maxfail=0 || upstream_spmd_status=$?

python -m pytest --import-mode=importlib --collect-only -q \
    "${repo_root}/python/cudf_polars/tests" > "${output_dir}/local-test-nodeids.txt"

python "${repo_root}/ci/analyze_cudf_polars_coverage.py" \
    --local-data "${output_dir}/.coverage.local" \
    --upstream-data "${output_dir}/.coverage.upstream" \
    --polars-version "${polars_version}" \
    --polars-sha "${polars_sha}" \
    --cudf-sha "${cudf_sha}" \
    --local-exit-status "${local_status}" \
    --upstream-in-memory-exit-status "${upstream_in_memory_status}" \
    --upstream-spmd-exit-status "${upstream_spmd_status}" \
    --gpu-run-list "${repo_root}/ci/cudf_polars/gpu_test_run_list.json" \
    --output-dir "${output_dir}"
python "${repo_root}/ci/classify_cudf_polars_tests.py" \
    --local-data "${output_dir}/.coverage.local" \
    --broad-data "${output_dir}/.coverage.upstream" \
    --tests-root "${repo_root}/python/cudf_polars/tests" \
    --collected-nodeids "${output_dir}/local-test-nodeids.txt" \
    --output-dir "${output_dir}"

echo "Wrote local/upstream coverage comparison and local-test classification reports to ${output_dir}"
if ((local_status || upstream_in_memory_status || upstream_spmd_status)); then
    echo "One or more test suites failed; see suite_exit_statuses in coverage-comparison.json." >&2
    exit 1
fi
