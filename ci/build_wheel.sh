#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2
shift 2

# Parse optional flags
stable_abi=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stable)
      stable_abi=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

source rapids-configure-sccache
source rapids-datetime-string
source rapids-init-pip

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}-${RAPIDS_CONDA_ARCH}-cuda${RAPIDS_CUDA_VERSION%%.*}-wheel-preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

RAPIDS_VERSION_SUFFIX=".post${RAPIDS_DATETIME_STRING}" \
  rapids-generate-version > ./VERSION

RAPIDS_VERSION_SUFFIX=".post${RAPIDS_DATETIME_STRING}" \
  rapids-generate-version > ./python/cudf/cudf/VERSION

cd "${package_dir}"

sccache --stop-server 2>/dev/null || true

rapids-logger "Building '${package_name}' wheel"

build_env_dir="/tmp/${package_name}-wheel-build-env"
# `build` preserves this environment after a failed build for debugging. CI
# retries must start with an empty directory, as required by `--env-dir`.
rm -rf "${build_env_dir}"

RAPIDS_BUILD_ARGS=(
  --wheel
  --outdir dist
  --verbose
  # A fixed location keeps isolated-build include paths stable for sccache.
  --env-dir "${build_env_dir}"
  --dependency-constraints-txt "${PIP_CONSTRAINT}"
)

# Add py-api setting for stable ABI builds
if [[ "${stable_abi}" == "true" ]] && [[ -n "${RAPIDS_PY_API:-}" ]]; then
  RAPIDS_BUILD_ARGS+=(--config-setting="skbuild.wheel.py-api=${RAPIDS_PY_API}")
fi

# `build` receives the same generated constraints explicitly. Unset the
# environment variable so it does not constrain the frontend installation.
unset PIP_CONSTRAINT

rapids-telemetry-record build-${package_name}.log python -m build \
    "${RAPIDS_BUILD_ARGS[@]}" \
    .

rapids-telemetry-record sccache-stats-${package_name}.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true
