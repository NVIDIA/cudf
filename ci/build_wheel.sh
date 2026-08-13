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

RAPIDS_PIP_WHEEL_ARGS=(
  -w dist
  -v
  --no-deps
  --disable-pip-version-check
)

# Add py-api setting for stable ABI builds
if [[ "${stable_abi}" == "true" ]] && [[ -n "${RAPIDS_PY_API:-}" ]]; then
  RAPIDS_PIP_WHEEL_ARGS+=(--config-settings="skbuild.wheel.py-api=${RAPIDS_PY_API}")
fi

# Only use --build-constraint when build isolation is enabled.
#
# Passing '--build-constraint' and '--no-build-isolation` together results in an error from 'pip',
# but we want to keep environment variable PIP_CONSTRAINT set unconditionally.
# PIP_NO_BUILD_ISOLATION=0 means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
if [[ "${PIP_NO_BUILD_ISOLATION:-}" != "0" ]]; then
    RAPIDS_PIP_WHEEL_ARGS+=(--build-constraint="${PIP_CONSTRAINT}")
fi

# unset PIP_CONSTRAINT (set by rapids-init-pip)... it doesn't affect builds as of pip 25.3, and
# results in an error from 'pip wheel' when set and --build-constraint is also passed
unset PIP_CONSTRAINT

rapids-telemetry-record build-${package_name}.log rapids-pip-retry wheel \
    "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
    .

rapids-telemetry-record sccache-stats-${package_name}.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true
