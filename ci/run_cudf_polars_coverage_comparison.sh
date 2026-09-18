#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare coverage from the local cudf-polars suite and the checked-out upstream
# Polars suite. Run ci/test_cudf_polars_polars_tests.sh first to prepare polars/.
set -euo pipefail

repo_root="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
output_dir="${1:-${repo_root}/coverage-results/cudf-polars-comparison}"
config="${repo_root}/ci/cudf_polars_coverage_analysis.toml"
polars_dir="${POLARS_DIR:-${repo_root}/polars}"

if [[ ! -d "${polars_dir}/py-polars/tests" ]]; then
    echo "Missing ${polars_dir}/py-polars/tests. Run ci/test_cudf_polars_polars_tests.sh first." >&2
    exit 2
fi

mkdir -p "${output_dir}"
# coverage.py leaves parallel data files alongside its combined output when
# --keep is used. Remove both forms so a rerun cannot mix stale test contexts
# into a new report.
rm -f "${output_dir}"/.coverage*

COVERAGE_FILE="${output_dir}/.coverage.local" \
    "${repo_root}/ci/run_cudf_polars_pytests.sh" \
    --numprocesses=4 --dist=worksteal \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" --cov-report=
COVERAGE_FILE="${output_dir}/.coverage.local" \
    python -m coverage combine --keep "${output_dir}"

# The broad runner performs the in-memory and SPMD runs. --cov-append keeps
# both engine contexts in the upstream data file.
POLARS_DIR="${polars_dir}" COVERAGE_FILE="${output_dir}/.coverage.upstream" \
    "${repo_root}/ci/run_cudf_polars_polars_tests.sh" \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" \
    --cov-report= --cov-append
COVERAGE_FILE="${output_dir}/.coverage.upstream" \
    python -m coverage combine --keep "${output_dir}"

python -m pytest --import-mode=importlib --collect-only -q \
    "${repo_root}/python/cudf_polars/tests" > "${output_dir}/local-test-nodeids.txt"

python "${repo_root}/ci/analyze_cudf_polars_coverage.py" \
    --local-data "${output_dir}/.coverage.local" \
    --upstream-data "${output_dir}/.coverage.upstream" \
    --output-dir "${output_dir}"
python "${repo_root}/ci/classify_cudf_polars_tests.py" \
    --local-data "${output_dir}/.coverage.local" \
    --broad-data "${output_dir}/.coverage.upstream" \
    --tests-root "${repo_root}/python/cudf_polars/tests" \
    --collected-nodeids "${output_dir}/local-test-nodeids.txt" \
    --output-dir "${output_dir}"

echo "Wrote coverage comparison and conservative local-test classification reports to ${output_dir}"
