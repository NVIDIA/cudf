#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

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
# Run all nvbench benchmarks with --profile and rmm_mode=cuda
for bench in *_NVBENCH; do
  if [[ -x "$bench" && -f "$bench" ]]; then
    start_time=$(date +%s)
    echo "Running $bench with --profile..."
    "./$bench" --profile --devices 0 -q --rmm_mode cuda
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

echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
