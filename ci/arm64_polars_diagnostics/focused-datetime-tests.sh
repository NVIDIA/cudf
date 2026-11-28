#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"
# Keep the original worker count while increasing the frequency of the two
# expressions that have reproduced the native fault.
exec timeout --kill-after=15s 4m python -m pytest -x --cache-clear -p no:benchmark     --numprocesses=4 --dist=worksteal     --junitxml="$CUDF_POLARS_DIAGNOSTICS_DIR/test-results/junit-datetime.xml"     tests/expressions/test_datetime_basic.py     -k '(test_datetime_truncate or test_datetime_offset_by) and (spmd or dask)'
