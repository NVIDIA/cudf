#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

evidence=artifacts/arm64-polars
mkdir -p "$evidence"
nvidia-smi --query-gpu=uuid,driver_version,pci.bus_id --format=csv > "$evidence/gpus.csv"

# Both historical shuffle faults occurred on this GPU. Keep one comparison
# from the normal pool to check whether the failure also occurs elsewhere.
target_gpu=GPU-ffa4d437-0552-ea7a-0e5d-0fe4bf4fd918
gpu_inventory=$(< "$evidence/gpus.csv")
repeat=false
if [[ "$gpu_inventory" == *"${target_gpu},"* ]]; then
    repeat=true
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == gdb-[0-9][0-9] ||
    "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == plain-[0-9][0-9] ]]; then
    echo 'Target GPU was not assigned' | tee "$evidence/skipped.txt"
    echo 'replay=false' >> "$GITHUB_OUTPUT"
    exit 0
fi

bash ci/arm64_polars_diagnostics/gpu-health.sh > "$evidence/gpu-health-before.txt" 2>&1
printf 'replay=true\nrepeat=%s\n' "$repeat" >> "$GITHUB_OUTPUT"
