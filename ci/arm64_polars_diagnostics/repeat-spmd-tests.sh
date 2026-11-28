#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-repeat-aggregation ]]; then
    cases=(
        'tests/expressions/test_agg.py::test_agg[nulls-max-Int8-unsorted-spmd-small]'
        'tests/expressions/test_agg.py::test_agg[nulls-max-Int8-sorted-spmd-small]'
    )
else
    cases=(
        'tests/streaming/test_join.py::test_join[spmd-small-options1-False-left]'
        'tests/streaming/test_unique.py::test_unique[spmd-small-none-subset15-False]'
    )
fi
# Each worker needs the same actor and allocator reuse under concurrent GPU use.
exec timeout --kill-after=15s 24m python -m pytest --cache-clear -p no:benchmark \
    --numprocesses=4 --dist=each -x \
    --junitxml="$RAPIDS_TESTS_DIR/junit-spmd-stress.xml" "${cases[@]}"
