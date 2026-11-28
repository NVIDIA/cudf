#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

sanitizer=(
    timeout --kill-after=15s 8m compute-sanitizer --tool memcheck
    --error-exitcode 86 --launch-timeout 120 --target-processes application-only --track-stream-ordered-races all
    --kernel-name kns=DeviceScanByKeyInitKernel --kernel-name kns=DeviceScanByKeyKernel
    --log-file "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck-packed-kernels.log"
)
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck-packed-native-dask-datetime ]]; then
    # Exercise this build's packed-size CUB scans without Python or Dask startup.
    compute-sanitizer --version
    exec "${sanitizer[@]}" "$CONDA_PREFIX/bin/gtests/libcudf/COPYING_TEST" \
        --gtest_filter=PackUnpackTest.SingleColumnFixedWidth:PackUnpackTest.SingleColumnFixedWidthNonNullable \
        --gtest_output="xml:$RAPIDS_TESTS_DIR/junit-packed-kernels.xml"
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" != memcheck-packed-polars-old-dask-datetime ]]; then
    python -m apply_ownership_fix "$CUDF_POLARS_OWNERSHIP_UTILS"
fi
args=(-m pytest -x -s -p no:benchmark
      --junitxml="$RAPIDS_TESTS_DIR/junit-packed-kernels.xml")
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck-packed-polars-* ]]; then
    # Include the actual crashing case and the no-op whose handoff shared buffers.
    args+=('tests/expressions/test_datetime_basic.py::test_datetime_truncate[spmd-small-dtype123-10m]'
           'tests/expressions/test_datetime_basic.py::test_datetime_truncate[spmd-small-dtype114-1us]')
else
    export CUDF_POLARS_SANITIZE_PACKED=1
    export CUDF_POLARS_REDUCED_ROUNDS=50
    args+=("test_chunk_handoff_reduction.py::test_chunk_handoff[alias]")
fi
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-order-packed-names-dask-datetime ]]; then
    export CUDF_POLARS_SINGLE_PROCESS_TRACE=1 CUDF_POLARS_KERNEL_NAMES=1
    exec timeout --kill-after=15s 6m python "${args[@]}"
fi
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == *-alternate-dask-datetime ]]; then
    export LD_LIBRARY_PATH="$CUDF_POLARS_ALTERNATE_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
compute-sanitizer --version
exec "${sanitizer[@]}" python "${args[@]}"
