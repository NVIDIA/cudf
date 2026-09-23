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

FORMAT_FILE_URL="https://raw.githubusercontent.com/rapidsai/rapids-cmake/${RAPIDS_BRANCH}/cmake-format-rapids-cmake.json"
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-format-rapids-cmake.json
mkdir -p "$(dirname "${RAPIDS_CMAKE_FORMAT_FILE}")"
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} "${FORMAT_FILE_URL}"

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
