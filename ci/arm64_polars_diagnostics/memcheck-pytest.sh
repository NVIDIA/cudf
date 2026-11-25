#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

compute-sanitizer --version
cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"

# The unfiltered run exhausted the job timeout during join. Check the shuffle
# kernels in each case separately so a stalled join still leaves time for unique.
cases=(
    'tests/streaming/test_join.py::test_join[spmd-small-options1-False-left]'
    'tests/streaming/test_unique.py::test_unique[spmd-small-none-subset15-False]'
)
result=0
for node in "${cases[@]}"; do
    name=${node%%::*}
    name=${name##*/}
    name=${name%.py}
    status=0
    timeout --signal=TERM --kill-after=15s 12m \
        compute-sanitizer --tool memcheck --error-exitcode 86 --launch-timeout 120 \
        --kernel-name kns=compute_row_partition_numbers \
        --kernel-name kns=copy_block_partitions \
        --kernel-name kns=copy_partitions \
        --log-file "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck-$name.log" \
        python -m pytest -v --cache-clear -p no:benchmark \
        --junitxml="$RAPIDS_TESTS_DIR/junit-$name.xml" "$node" || status=$?
    printf '%s\t%s\n' "$name" "$status" >> "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck-status.tsv"
    if [[ "$status" != 0 ]]; then
        result=$status
    fi
done
exit "$result"
