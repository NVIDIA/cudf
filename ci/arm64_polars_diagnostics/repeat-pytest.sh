#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

attempts=(1 2 3)
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-paired-dask-datetime ]]; then
    # Start fresh pytest and Dask processes for each condition on the same GPU.
    attempts=(call reset)
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
    if [[ "$attempt" == call || "$attempt" == reset ]]; then
        trial="signal-${attempt}-dask-datetime"
        printf '%s\n' "$trial" > "$output/trial.txt"
        echo "Full-suite GPU replay with Dask condition $attempt"
    else
        echo "Full-suite GPU replay $attempt of 3"
    fi
    CUDF_POLARS_DIAGNOSTICS_DIR="$output" \
        CUDF_POLARS_DIAGNOSTICS_TRIAL="$trial" \
        "$(dirname "${BASH_SOURCE[0]}")/run_cudf_polars_pytests-once.sh" "${args[@]}" || {
        status=$?
        # Preserve the other condition even if this one reproduced the crash.
        if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" != signal-paired-dask-datetime ]]; then
            exit "$status"
        fi
    }
done
exit "$status"
