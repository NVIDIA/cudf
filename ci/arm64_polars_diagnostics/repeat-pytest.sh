#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

for attempt in 1 2 3; do
    output="$CUDF_POLARS_DIAGNOSTICS_DIR/attempt-$attempt"
    mkdir -p "$output"/{test-results,coverage}
    args=()
    for arg in "$@"; do
        args+=("${arg//"$CUDF_POLARS_DIAGNOSTICS_DIR"/"$output"}")
    done
    echo "Full-suite GPU replay $attempt of 3"
    CUDF_POLARS_DIAGNOSTICS_DIR="$output" "$(dirname "${BASH_SOURCE[0]}")/run_cudf_polars_pytests-once.sh" "${args[@]}"
done
