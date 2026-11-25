#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

evidence=artifacts/arm64-polars
mkdir -p "$evidence"
nvidia-smi --query-gpu=uuid,driver_version,pci.bus_id --format=csv > "$evidence/gpus.csv"

# Repeat on the GPU associated with the failure being replayed.
target_gpu=GPU-ffa4d437-0552-ea7a-0e5d-0fe4bf4fd918
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == *-aggregation ]]; then
    target_gpu=GPU-55411e83-90e6-d36b-43de-0d7e24c8489a
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == *-dask-datetime ]]; then
    target_gpu=GPU-1ae60e36-999e-c6da-6055-0ad23bd269d2
fi
gpu_inventory=$(< "$evidence/gpus.csv")
repeat=false
if [[ "$gpu_inventory" == *"${target_gpu},"* ]]; then
    repeat=true
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == gdb-[0-9][0-9] ||
    "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == plain-[0-9][0-9] ||
    "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-target-*-aggregation ||
    "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-target-*-dask-datetime ||
    "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-context-target-*-frozen-shuffle ]]; then
    echo 'Target GPU was not assigned' | tee "$evidence/skipped.txt"
    echo 'replay=false' >> "$GITHUB_OUTPUT"
    exit 0
fi

bash ci/arm64_polars_diagnostics/gpu-health.sh > "$evidence/gpu-health-before.txt" 2>&1
printf 'replay=true\nrepeat=%s\n' "$repeat" >> "$GITHUB_OUTPUT"
