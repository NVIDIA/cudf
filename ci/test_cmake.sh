#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cmake.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

rapids-logger "Create CMake test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cmake \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n cmake_tests

# Temporarily allow unbound variables for conda activation.
set +u
conda activate cmake_tests
set -u

rapids-print-env

rapids-logger "Run cuDF CMake tests"
cmake -S cpp/cmake/tests \
      -B cpp/build/cmake-tests \
      -GNinja \
      -DCUDF_REPOSITORY_DIR="${PWD}"
ctest --test-dir cpp/build/cmake-tests --output-on-failure -j"$(nproc)"
