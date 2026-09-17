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
strict_data="${output_dir}/.coverage.strict"

if [[ ! -d "${polars_dir}/py-polars/tests" ]]; then
    echo "Missing ${polars_dir}/py-polars/tests. Run ci/test_cudf_polars_polars_tests.sh first." >&2
    exit 2
fi

mkdir -p "${output_dir}"
rm -f "${output_dir}/.coverage.local" "${output_dir}/.coverage.upstream" "${strict_data}"

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

# Capture the no-fallback subset separately from the broad compatibility data.
# The broad runner includes this subset, but its combined coverage cannot prove
# that a local test is covered without fallback.
(
    cd "${polars_dir}"
    COVERAGE_FILE="${strict_data}" python -m pytest \
        --import-mode=importlib \
        -p cudf_polars.testing.inject_gpu_engine \
        --inject-gpu-engine in-memory \
        --inject-gpu-engine-raise-on-fail \
        --cov=cudf_polars --cov-context=test --cov-config="${config}" --cov-report= \
        py-polars/tests
    CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE=805306368 \
    CUDF_POLARS__EXECUTOR__FALLBACK_MODE=silent \
    COVERAGE_FILE="${strict_data}" python -m pytest \
        --import-mode=importlib \
        -p cudf_polars.testing.inject_gpu_engine \
        --inject-gpu-engine spmd \
        --inject-gpu-engine-blocksize small \
        --inject-gpu-engine-raise-on-fail \
        --cov=cudf_polars --cov-context=test --cov-config="${config}" --cov-report= \
        --cov-append py-polars/tests
)
COVERAGE_FILE="${strict_data}" python -m coverage combine --keep "${output_dir}"

python -m pytest --import-mode=importlib --collect-only -q \
    "${repo_root}/python/cudf_polars/tests" > "${output_dir}/local-test-nodeids.txt"

python "${repo_root}/ci/analyze_cudf_polars_coverage.py" \
    --local-data "${output_dir}/.coverage.local" \
    --upstream-data "${output_dir}/.coverage.upstream" \
    --output-dir "${output_dir}"
python "${repo_root}/ci/classify_cudf_polars_tests.py" \
    --local-data "${output_dir}/.coverage.local" \
    --broad-data "${output_dir}/.coverage.upstream" \
    --strict-data "${strict_data}" \
    --tests-root "${repo_root}/python/cudf_polars/tests" \
    --collected-nodeids "${output_dir}/local-test-nodeids.txt" \
    --output-dir "${output_dir}"

echo "Wrote coverage comparison, strict GPU candidates, and local-test classification reports to ${output_dir}"
