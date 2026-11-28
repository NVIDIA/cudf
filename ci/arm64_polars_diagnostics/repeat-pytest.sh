#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

attempts=(1 2 3)
handoff_mode=
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-paired-dask-datetime ]]; then
    # Start fresh pytest and Dask processes for each condition on the same GPU.
    attempts=(call reset)
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-driver-dask-datetime ]]; then
    attempts=(baseline alternate)
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-order-alternate-dask-datetime ]]; then
    attempts=(alternate alternate-repeat)
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-order-focused*-dask-datetime ]]; then
    attempts=({1..20})
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-order-handoff-* ]]; then
    handoff_mode=${CUDF_POLARS_DIAGNOSTICS_TRIAL#signal-order-handoff-}
    handoff_mode=${handoff_mode%%-*}
    attempts=()
    pairs=20
    if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == *-full-dask-datetime ]]; then
        pairs=3
    fi
    for (( round = 1; round <= pairs; round++ )); do
        # Counterbalance which condition sees the GPU first.
        if (( round % 2 )); then
            attempts+=("observe-$round" "$handoff_mode-$round")
        else
            attempts+=("$handoff_mode-$round" "observe-$round")
        fi
    done
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
    if [[ -n "$handoff_mode" ]]; then
        condition=${attempt%-*}
        environment+=("CUDF_POLARS_HANDOFF_MODE=$condition")
        printf '%s\n' "$condition" > "$output/condition.txt"
        echo "Paired GPU replay $attempt"
    elif [[ "$attempt" == call || "$attempt" == reset ]]; then
        trial="signal-${attempt}-dask-datetime"
        printf '%s\n' "$trial" > "$output/trial.txt"
        echo "Full-suite GPU replay with Dask condition $attempt"
    elif [[ "$attempt" == baseline || "$attempt" == alternate* ]]; then
        expected=libcuda.so.615.71.10
        if [[ "$attempt" == alternate* ]]; then
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
    elif [[ "$trial" == signal-order-focused*-dask-datetime ]]; then
        if [[ "$trial" == signal-order-focused-lifetime-dask-datetime ]]; then
            environment+=("CUDF_POLARS_HANDOFF_MODE=lifetime")
        fi
        echo "Datetime GPU replay $attempt of ${#attempts[@]}"
    else
        echo "Full-suite GPU replay $attempt of ${#attempts[@]}"
    fi
    run_status=0
    env "${environment[@]}" CUDF_POLARS_DIAGNOSTICS_TRIAL="$trial" \
        "$(dirname "${BASH_SOURCE[0]}")/run_cudf_polars_pytests-once.sh" "${args[@]}" ||
        run_status=$?
    printf '%s\n' "$run_status" > "$output/exit-status.txt"
    if (( run_status )); then
        status=$run_status
        # Only paired comparisons need to continue after a crash.
        if [[ "$attempt" =~ ^[0-9]+$ || "$trial" == signal-order-alternate-dask-datetime ]]; then
            exit "$status"
        fi
    fi
done
exit "$status"
