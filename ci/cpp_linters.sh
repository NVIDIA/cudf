#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n clang_tidy

# Temporarily allow unbound variables for conda activation.
set +u
conda activate clang_tidy
set -u

# clang-tidy parses the GCC compile command with clang. Newer conda compilers add
# this GCC-only optimization flag, which clang reports as an error.
for flags_var in CFLAGS CXXFLAGS; do
  if [[ -n "${!flags_var:-}" ]]; then
    export "${flags_var}=$(printf '%s' "${!flags_var}" | sed -E 's/(^|[[:space:]])-fno-merge-constants([[:space:]]|$)/ /g; s/[[:space:]]+/ /g; s/^ //; s/ $//')"
  fi
done

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="cudf-cpp-linters-preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

source rapids-configure-sccache

sccache --stop-server 2>/dev/null || true

# Run the build via CMake, which will run clang-tidy when CUDF_STATIC_LINTERS is enabled.

iwyu_flag=""
# Temporarily disabling this until we can figure out why it causes the build to hang
# if [[ "${RAPIDS_BUILD_TYPE:-}" == "nightly" ]]; then
#  iwyu_flag="-DCUDF_IWYU=ON"
# fi
rapids-telemetry-record cpp_linters_build.log cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCUDF_CLANG_TIDY=ON ${iwyu_flag} -DBUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=75 -GNinja
cmake --build cpp/build 2>&1 | python cpp/scripts/parse_iwyu_output.py

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# Remove invalid components of the path for local usage. The path below is
# valid in the CI due to where the project is cloned, but presumably the fixes
# will be applied locally from inside a clone of cudf.
sed -i 's/\/__w\/cudf\/cudf\///' iwyu_results.txt
