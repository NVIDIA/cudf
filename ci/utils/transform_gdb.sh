#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# CTest preloads its stream checker into the test, not into the debugger.
test_preload="${LD_PRELOAD:-}"
unset LD_PRELOAD

exec gdb --batch --return-child-result \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "set disable-randomization off" \
    -ex "set debuginfod enabled off" \
    -ex "set print elements 32" \
    -ex "set startup-with-shell off" \
    -ex "set environment LD_PRELOAD ${test_preload}" \
    -ex run \
    -ex "thread apply all bt full" \
    -ex "info sharedlibrary" \
    --args "${TRANSFORM_GDB_BINARY:-${BASH_SOURCE[0]}.diagnostic-real}" "$@"
