#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# CI provides LIBCUDF_KERNEL_CACHE_PATH through the reusable workflow's cache-environment input.
# Resolve a workspace-relative value before CTest changes its working directory.
LIBCUDF_KERNEL_CACHE_PATH="${LIBCUDF_KERNEL_CACHE_PATH:-.cache/libcudf}"
if [[ "${LIBCUDF_KERNEL_CACHE_PATH}" != /* ]]; then
  LIBCUDF_KERNEL_CACHE_PATH="$(realpath -m "${LIBCUDF_KERNEL_CACHE_PATH}")"
fi
export LIBCUDF_KERNEL_CACHE_PATH
mkdir -p "${LIBCUDF_KERNEL_CACHE_PATH}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi
