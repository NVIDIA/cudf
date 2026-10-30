#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds libcudf and its gtests from source with CMAKE_BUILD_TYPE=Debug and runs the gtests.
# Must be run on a GPU node since the build targets the native GPU architecture.

set -euo pipefail

# Support invoking test_cpp_debug.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

rapids-logger "Create debug build conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp_debug \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test_cpp_debug

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test_cpp_debug
set -u

# Keep the preprocessor cache separate from the Release conda builds since the flags never match.
SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="libcudf-$(arch)-cuda${RAPIDS_CUDA_VERSION%%.*}-debug-preprocessor-cache"
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

source rapids-configure-sccache

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

sccache --stop-server 2>/dev/null || true

rapids-logger "Build libcudf and gtests (Debug)"
# -g: Debug build, -n: do not install, benchmarks are not built unless requested.
# The GPU architecture defaults to the native GPU on this node.
./build.sh -g -n libcudf tests

rapids-logger "Configured build settings"
grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|BUILD_TESTS|BUILD_BENCHMARKS):" cpp/build/CMakeCache.txt

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run libcudf gtests (Debug)"
# Debug host code is unoptimized, so allow more time than ci/test_cpp.sh does.
timeout 2h ctest --test-dir cpp/build --output-on-failure --no-tests=error -j20

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
