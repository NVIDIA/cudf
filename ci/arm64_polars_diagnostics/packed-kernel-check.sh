#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"
python -m apply_ownership_fix "$CUDF_POLARS_OWNERSHIP_UTILS"
export CUDF_POLARS_SANITIZE_PACKED=1
export CUDF_POLARS_REDUCED_ROUNDS=50
args=(-m pytest -x -s -p no:benchmark
      "test_chunk_handoff_reduction.py::test_chunk_handoff[alias]"
      --junitxml="$RAPIDS_TESTS_DIR/junit-packed-kernels.xml")
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-order-packed-names-dask-datetime ]]; then
    export CUDF_POLARS_SINGLE_PROCESS_TRACE=1 CUDF_POLARS_KERNEL_NAMES=1
    exec timeout --kill-after=15s 6m python "${args[@]}"
fi
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == *-alternate-dask-datetime ]]; then
    export LD_LIBRARY_PATH="$CUDF_POLARS_ALTERNATE_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
compute-sanitizer --version
exec timeout --kill-after=15s 8m compute-sanitizer --tool memcheck \
    --error-exitcode 86 --launch-timeout 120 --track-stream-ordered-races all \
    --kernel-name kns=DeviceScanByKeyInitKernel --kernel-name kns=DeviceScanByKeyKernel \
    --log-file "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck-packed-kernels.log" python "${args[@]}"
