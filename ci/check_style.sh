#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n checks
conda activate checks

RAPIDS_BRANCH="$(cat "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../RAPIDS_BRANCH)"

export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-format-rapids-cmake.json
mkdir -p "$(dirname "${RAPIDS_CMAKE_FORMAT_FILE}")"
rapids-github-fetch rapidsai/rapids-cmake cmake-format-rapids-cmake.json "${RAPIDS_BRANCH}" > "${RAPIDS_CMAKE_FORMAT_FILE}"

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
