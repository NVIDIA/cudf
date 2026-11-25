#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

evidence=artifacts/arm64-polars
mkdir -p "$evidence"
nvidia-smi --query-gpu=uuid,driver_version,pci.bus_id --format=csv > "$evidence/gpus.csv"

trial=$CUDF_POLARS_DIAGNOSTICS_TRIAL
gpu_inventory=$(< "$evidence/gpus.csv")
shuffle_gpu=GPU-ffa4d437-0552-ea7a-0e5d-0fe4bf4fd918
dask_gpus=(
    GPU-1ae60e36-999e-c6da-6055-0ad23bd269d2
    GPU-72a129d8-0178-3e41-9997-3c8f9c9b46ae
    "$shuffle_gpu"
)
if [[ "$trial" == probe-* ]]; then
    # Both backend failures have occurred on the shuffle GPU.
    if [[ "$gpu_inventory" == *"$shuffle_gpu,"* && "$trial" == *[02468] ]]; then
        trial=signal-context-target-frozen-shuffle
    else
        for gpu in "${dask_gpus[@]}"; do
            if [[ "$gpu_inventory" == *"$gpu,"* ]]; then
                trial=signal-target-dask-datetime
                break
            fi
        done
    fi
fi

# Repeat on GPUs associated with the failures being replayed.
target_gpus=("$shuffle_gpu")
if [[ "$trial" == *-aggregation ]]; then
    target_gpus=(GPU-55411e83-90e6-d36b-43de-0d7e24c8489a)
elif [[ "$trial" == *-dask-datetime ]]; then
    target_gpus=("${dask_gpus[@]}")
fi
repeat=false
for target_gpu in "${target_gpus[@]}"; do
    if [[ "$gpu_inventory" == *"${target_gpu},"* ]]; then
        repeat=true
        break
    fi
done
if [[ "$repeat" == false &&
    ( "$trial" == probe-* ||
        "$trial" == gdb-[0-9][0-9] ||
        "$trial" == plain-[0-9][0-9] ||
        "$trial" == signal-target-*-aggregation ||
        "$trial" == signal-target-*-dask-datetime ||
        "$trial" == signal-context-target-*-frozen-shuffle ) ]]; then
    echo 'Target GPU was not assigned' | tee "$evidence/skipped.txt"
    echo 'replay=false' >> "$GITHUB_OUTPUT"
    exit 0
fi

bash ci/arm64_polars_diagnostics/gpu-health.sh > "$evidence/gpu-health-before.txt" 2>&1
printf '%s\n' "$trial" > "$evidence/trial.txt"
printf 'replay=true\nrepeat=%s\ntrial=%s\n' "$repeat" "$trial" >> "$GITHUB_OUTPUT"
