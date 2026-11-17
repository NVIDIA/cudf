#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp_benchmarks.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_cpp_common.sh

rapids-logger "Run tests of libcudf benchmarks"
./ci/run_cudf_benchmark_smoketests.sh
