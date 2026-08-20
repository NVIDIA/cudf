#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.
# Support invoking run_cudf_polars_pytests.sh outside the script directory
TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

python "${TIMEOUT_TOOL_PATH}" --enable-python 5400 \
       python -m pytest --cache-clear "$@" tests
