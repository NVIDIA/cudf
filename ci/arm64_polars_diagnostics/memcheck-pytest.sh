#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

compute-sanitizer --version
cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"

# A single SPMD process gives memcheck the CUDA context used by both callers.
exec compute-sanitizer --tool memcheck --error-exitcode 86 --launch-timeout 120 \
    --log-file "$CUDF_POLARS_DIAGNOSTICS_DIR/memcheck.log" \
    python -m pytest -v --cache-clear -p no:benchmark \
    --junitxml="$RAPIDS_TESTS_DIR/junit-cudf-polars.xml" \
    'tests/streaming/test_join.py::test_join[spmd-small-options1-False-left]' \
    'tests/streaming/test_unique.py::test_unique[spmd-small-none-subset15-False]'
