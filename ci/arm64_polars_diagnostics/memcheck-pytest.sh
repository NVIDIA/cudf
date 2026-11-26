#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

compute-sanitizer --version
sanitizer=(
    timeout --signal=TERM --kill-after=15s 12m
    compute-sanitizer --tool memcheck --error-exitcode 86 --launch-timeout 120
)
# Both crash paths call packed_size. Its transform, reduce, and scan kernels
# are CUB template instantiations with these names in the archived cubin.
packed_size_filters=(
    --kernel-name kns=compute_splits
    --kernel-name kns=buf_size_functor
    --kernel-name kns=split_key_functor
)
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck-native-aggregation ]]; then
    # The Python runs stalled during SPMD fixture startup. Exercise the same
    # native packing path with a packaged streaming gtest.
    exec "${sanitizer[@]}" "${packed_size_filters[@]}" \
        --log-file "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck-native.log" \
        "$CONDA_PREFIX/bin/gtests/libcudf_streaming/libcudf_streaming_single_tests" \
        --gtest_filter=Partitions/NumOfPartitions.round_trip_with_reservations/13 \
        --gtest_output="xml:$RAPIDS_TESTS_DIR/junit-packed-size.xml"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"

# The unfiltered run exhausted the job timeout during join. Check the shuffle
# kernels in each case separately so a stalled join still leaves time for unique.
cases=(
    'tests/streaming/test_join.py::test_join[spmd-small-options1-False-left]'
    'tests/streaming/test_unique.py::test_unique[spmd-small-none-subset15-False]'
)
kernel_filters=(
    --kernel-name kns=compute_row_partition_numbers
    --kernel-name kns=copy_block_partitions
    --kernel-name kns=copy_partitions
)
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck-aggregation ]]; then
    cases=('tests/expressions/test_agg.py::test_agg[nulls-max-Int8-unsorted-spmd-small]')
    kernel_filters=("${packed_size_filters[@]}")
fi
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck-order-dask-datetime ]]; then
    # Preserve allocation, event, and use edges across kernels and child workers.
    kernel_filters=(--track-stream-ordered-races all --target-processes all)
    cases=(
        'tests/expressions/test_datetime_basic.py::test_datetime_truncate[spmd-small-dtype42-25h]'
        'tests/expressions/test_datetime_basic.py::test_datetime_offset_by[dask-dtype18-3us]'
    )
fi
result=0
for node in "${cases[@]}"; do
    name=${node##*::}
    name=${name//\[/_}
    name=${name//\]/}
    status=0
    "${sanitizer[@]}" "${kernel_filters[@]}" \
        --log-file "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck-$name.log" \
        python -m pytest -v --cache-clear -p no:benchmark \
        --junitxml="$RAPIDS_TESTS_DIR/junit-$name.xml" "$node" || status=$?
    printf '%s\t%s\n' "$name" "$status" >> "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck-status.tsv"
    if [[ "$status" != 0 ]]; then
        result=$status
    fi
done
exit "$result"
