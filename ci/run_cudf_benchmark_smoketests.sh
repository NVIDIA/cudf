#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."

# Support customizing the benchmarks' install location
# First, try the installed location (CI/conda environments)
installed_benchmark_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/libcudf/"
# Fall back to the build directory (devcontainer environments)
devcontainers_benchmark_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/benchmarks/"

if [[ -d "${installed_benchmark_location}" ]]; then
    cd "${installed_benchmark_location}"
elif [[ -d "${devcontainers_benchmark_location}" ]]; then
    cd "${devcontainers_benchmark_location}"
else
    echo "Error: Benchmark location not found. Searched:" >&2
    echo "  - ${installed_benchmark_location}" >&2
    echo "  - ${devcontainers_benchmark_location}" >&2
    exit 1
fi

EXITCODE=0
validation_dir="$(mktemp -d)"
trap 'rm -rf "${validation_dir}"' EXIT
# Run all nvbench benchmarks with --profile and rmm_mode=cuda
for bench in *_NVBENCH; do
  if [[ -x "$bench" && -f "$bench" ]]; then
    start_time=$(date +%s)
    echo "Running $bench with --profile..."
    args=(--profile --devices 0 -q --rmm_mode cuda)
    if [[ "$bench" == NDSH_* ]]; then
      args+=(--axis scale_factor=0.01)
      if [[ "$bench" =~ ^NDSH_Q(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|21)_NVBENCH$ ]]; then
        args+=(--output_directory "${validation_dir}/q${BASH_REMATCH[1]}")
      fi
    fi
    "./$bench" "${args[@]}"
    SUITEERROR=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    if (( SUITEERROR == 0 )); then
      echo "Benchmark $bench passed in $duration seconds"
    else
      echo "Benchmark $bench failed in $duration seconds: $SUITEERROR"
      EXITCODE=$SUITEERROR
    fi
  fi
done

python "${repo_root}/ci/validate_ndsh_benchmarks.py" \
  --output-dir "${validation_dir}" \
  --sql-dir "${repo_root}/cpp/libcudf_streaming/benchmarks/streaming/ndsh/sql"

echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
