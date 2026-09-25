#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Produce per-node CPU-fallback diagnostics for the checked-out upstream Polars
# suite. Run ci/test_cudf_polars_polars_tests.sh first to prepare polars/.
set -euo pipefail

repo_root="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
output_dir="${1:-${repo_root}/test-results/cudf-polars-fallback-diagnostics}"
polars_dir="${POLARS_DIR:-${repo_root}/polars}"

if [[ ! -d "${polars_dir}/py-polars/tests" ]]; then
    echo "Missing ${polars_dir}/py-polars/tests. Run ci/test_cudf_polars_polars_tests.sh first." >&2
    exit 2
fi

mkdir -p "${output_dir}"
in_memory_report="$(realpath -m "${output_dir}/in-memory.xml")"
spmd_report="$(realpath -m "${output_dir}/spmd-small-blocksize.xml")"

exit_code=0
POLARS_DIR="${polars_dir}" "${repo_root}/ci/run_cudf_polars_polars_tests.sh" \
    --engine in-memory --junitxml="${in_memory_report}" || exit_code=$?
POLARS_DIR="${polars_dir}" "${repo_root}/ci/run_cudf_polars_polars_tests.sh" \
    --engine spmd --inject-gpu-engine-blocksize small \
    --junitxml="${spmd_report}" || exit_code=$?

python "${repo_root}/ci/analyze_cudf_polars_fallbacks.py" \
    --report "in-memory=${in_memory_report}" \
    --report "spmd-small-blocksize=${spmd_report}" \
    --output-dir "${output_dir}"

echo "Wrote upstream Polars fallback diagnostics to ${output_dir}"
exit "${exit_code}"
