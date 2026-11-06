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

polars_version=$(python - "${repo_root}/python/cudf_polars/pyproject.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from polars import __version__

dependencies = tomllib.loads(Path(sys.argv[1]).read_text())["project"]["dependencies"]
requirement = next(Requirement(dep) for dep in dependencies if Requirement(dep).name == "polars")
if __version__ not in requirement.specifier:
    raise SystemExit(f"Installed Polars {__version__} does not satisfy {requirement}")
print(__version__)
PY
)
polars_sha=$(git -C "${polars_dir}" rev-parse HEAD)
polars_tag=$(git -C "${polars_dir}" describe --tags --exact-match HEAD 2>/dev/null || true)
if [[ "${polars_tag}" != "py-${polars_version}" ]]; then
    echo "Expected the Polars checkout at ${polars_dir} to be py-${polars_version}, got ${polars_tag:-an untagged revision}." >&2
    exit 2
fi
cudf_sha=$(git -C "${repo_root}" rev-parse HEAD)
collection_metadata=$(python - "${repo_root}" <<'PY'
import json
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(json.dumps({
    "architecture": platform.machine(),
    "local": {
        "data": ".coverage.local",
        "engine": "local test fixtures",
        "runner": "ci/run_cudf_polars_pytests.sh",
        "ignored_paths": ["tests/streaming/test_tpch.py", "tests/streaming/test_tpcds.py"],
    },
    "upstream": {
        "runner": "ci/run_cudf_polars_polars_tests.sh",
        "configurations": [
            {"data": ".coverage.upstream-in-memory", "engine": "in-memory", "blocksize": None},
            {"data": ".coverage.upstream-spmd", "engine": "spmd", "blocksize": "default"},
        ],
        "marker": "not slow",
        "gpu_run_list": None,
        "deselection_tables": {
            str(path.relative_to(root)): path.read_text().splitlines()
            for path in sorted((root / "ci/cudf_polars").glob("polars_test_deselections*.txt"))
        },
        "deselection_policy": "The runner applies architecture- and glibc-specific tables conditionally.",
    },
    "pytest_options": ["--numprocesses=4", "--dist=worksteal", "--cov-context=test"],
}))
PY
)

mkdir -p "${output_dir}"
rm -f "${output_dir}/.coverage.local" \
    "${output_dir}/.coverage.upstream-in-memory" \
    "${output_dir}/.coverage.upstream-spmd" \
    "${output_dir}/coverage-comparison.json"

local_status=0
COVERAGE_FILE="${output_dir}/.coverage.local" \
    "${repo_root}/ci/run_cudf_polars_pytests.sh" \
    --numprocesses=4 --dist=worksteal \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" --cov-report= || local_status=$?

# Keep engine data separate so later analysis can distinguish configuration
# gaps from missing upstream assertions, even when test node IDs are identical.
upstream_in_memory_status=0
CUDF_POLARS_GPU_RUN_LIST="" POLARS_DIR="${polars_dir}" \
COVERAGE_FILE="${output_dir}/.coverage.upstream-in-memory" \
    "${repo_root}/ci/run_cudf_polars_polars_tests.sh" --engine in-memory \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" \
    --cov-report= --maxfail=0 || upstream_in_memory_status=$?
upstream_spmd_status=0
CUDF_POLARS_GPU_RUN_LIST="" POLARS_DIR="${polars_dir}" \
COVERAGE_FILE="${output_dir}/.coverage.upstream-spmd" \
    "${repo_root}/ci/run_cudf_polars_polars_tests.sh" --engine spmd \
    --cov=cudf_polars --cov-context=test --cov-config="${config}" \
    --cov-report= --maxfail=0 || upstream_spmd_status=$?

python "${repo_root}/ci/analyze_cudf_polars_coverage.py" \
    --local-data "${output_dir}/.coverage.local" \
    --upstream-data "${output_dir}/.coverage.upstream-in-memory" \
    --upstream-data "${output_dir}/.coverage.upstream-spmd" \
    --polars-version "${polars_version}" \
    --polars-sha "${polars_sha}" \
    --cudf-sha "${cudf_sha}" \
    --collection-metadata "${collection_metadata}" \
    --local-exit-status "${local_status}" \
    --upstream-in-memory-exit-status "${upstream_in_memory_status}" \
    --upstream-spmd-exit-status "${upstream_spmd_status}" \
    --output-dir "${output_dir}"

echo "Wrote coverage data and line-level comparison to ${output_dir}"
if ((local_status || upstream_in_memory_status || upstream_spmd_status)); then
    echo "One or more test suites failed; see suite_exit_statuses in coverage-comparison.json." >&2
    exit 1
fi
