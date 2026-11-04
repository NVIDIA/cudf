#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.
# Support invoking run_cudf_polars_pytests.sh outside the script directory
TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

test_timeout=5400
if [[ ${RAPIDS_BUILD_TYPE:-} == pull-request ]]; then
    test_timeout=1800
fi

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

python "${TIMEOUT_TOOL_PATH}" --enable-python "${test_timeout}" \
       python -m pytest --cache-clear -p no:benchmark "$@" tests \
           --ignore=tests/streaming/test_tpch.py \
           --ignore=tests/streaming/test_tpcds.py
