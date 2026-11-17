#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_cpp_common.sh

rapids-logger "Memcheck gtests with rmm_mode=cuda"

timeout 4h ./ci/run_cudf_memcheck_ctests.sh && EXITCODE=$? || EXITCODE=$?;

rapids-logger "Test script exiting with value: $EXITCODE"
# shellcheck disable=SC2086
exit ${EXITCODE}
