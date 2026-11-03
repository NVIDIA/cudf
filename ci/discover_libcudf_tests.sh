#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-logger "Create test environment"
rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Discovering libcudf test executables"

# Navigate to test installation directory
TEST_DIR="${CONDA_PREFIX}/bin/gtests/libcudf"

if [ ! -d "${TEST_DIR}" ]; then
  rapids-logger "Error: Test directory ${TEST_DIR} not found"
  exit 1
fi

cd "${TEST_DIR}"

# Find all *_TEST executables
if ! ls *_TEST 1> /dev/null 2>&1; then
  rapids-logger "Error: No test executables found matching *_TEST pattern"
  exit 1
fi

# Create JSON array of test names (excluding STREAM_ tests)
test_array=()
for test in *_TEST; do
  if [[ ! "$test" =~ ^STREAM_ ]]; then
    test_array+=("$test")
  fi
done
tests=$(printf '%s\n' "${test_array[@]}" | jq -R -s -c 'split("\n") | map(select(length > 0))')

rapids-logger "Found tests:"
echo "${tests}" | jq .[]

# Output to GITHUB_OUTPUT for GitHub Actions
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "tests=${tests}" >> "${GITHUB_OUTPUT}"
fi
