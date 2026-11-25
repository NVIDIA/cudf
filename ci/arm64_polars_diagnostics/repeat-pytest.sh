#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

attempts=(1 2 3)
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-paired-dask-datetime ]]; then
    # Start fresh pytest and Dask processes for each condition on the same GPU.
    attempts=(call reset)
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-driver-dask-datetime ]]; then
    attempts=(baseline alternate)
fi
status=0
for attempt in "${attempts[@]}"; do
    output="$CUDF_POLARS_DIAGNOSTICS_DIR/attempt-$attempt"
    mkdir -p "$output"/{test-results,coverage}
    args=()
    for arg in "$@"; do
        args+=("${arg//"$CUDF_POLARS_DIAGNOSTICS_DIR"/"$output"}")
    done
    trial=$CUDF_POLARS_DIAGNOSTICS_TRIAL
    environment=("CUDF_POLARS_DIAGNOSTICS_DIR=$output")
    if [[ "$attempt" == call || "$attempt" == reset ]]; then
        trial="signal-${attempt}-dask-datetime"
        printf '%s\n' "$trial" > "$output/trial.txt"
        echo "Full-suite GPU replay with Dask condition $attempt"
    elif [[ "$attempt" == baseline || "$attempt" == alternate ]]; then
        expected=libcuda.so.615.71.10
        if [[ "$attempt" == alternate ]]; then
            expected=libcuda.so.620.70.10
            environment+=("LD_LIBRARY_PATH=$CUDF_POLARS_ALTERNATE_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
        fi
        echo "Full-suite GPU replay with CUDA driver library $expected"
        env "${environment[@]}" python - "$expected" > "$output/driver.txt" <<'PY'
import ctypes
import sys
from pathlib import Path

driver = ctypes.CDLL("libcuda.so.1")
paths = {
    Path(line.split()[-1])
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "/libcuda.so." in line
}
print("CUDA driver libraries:", *sorted(paths))
if {path.name for path in paths} != {sys.argv[1]}:
    raise SystemExit(f"Expected {sys.argv[1]}")
version = ctypes.c_int()
if driver.cuInit(0) != 0 or driver.cuDriverGetVersion(ctypes.byref(version)) != 0:
    raise SystemExit("Could not initialize the CUDA driver")
print("CUDA driver API version:", version.value)
PY
        cat "$output/driver.txt"
    else
        echo "Full-suite GPU replay $attempt of 3"
    fi
    env "${environment[@]}" CUDF_POLARS_DIAGNOSTICS_TRIAL="$trial" \
        "$(dirname "${BASH_SOURCE[0]}")/run_cudf_polars_pytests-once.sh" "${args[@]}" || {
        status=$?
        # Preserve the other condition even if this one reproduced the crash.
        if [[ "$attempt" == [123] ]]; then
            exit "$status"
        fi
    }
done
exit "$status"
