#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container build of a static libcudf install tree.
#
# Expects a RAPIDS ci-wheel environment. Sources the toolchain from
# setup_java_env.sh, builds libcudf with BUILD_SHARED_LIBS=OFF, and installs
# the resulting static libcudf (plus its static dependencies) into
# INSTALL_PREFIX. When HOST_UID / HOST_GID are set, chowns the install tree so
# the host user owns the outputs.
#
# Inputs (environment variables):
#   RAPIDS_CUDA_VERSION        CUDA version, e.g. 12.9.2 (required).
#   PARALLEL_LEVEL             Build parallelism (default: nproc).
#   CMAKE_CUDA_ARCHITECTURES   Optional override for -DCMAKE_CUDA_ARCHITECTURES.
#   INSTALL_PREFIX             Static libcudf install dir (default: /output).
#   REPO_ROOT                  cuDF checkout (default: /repo).
#   BUILD_DIR                  CMake build dir (default: /tmp/libcudf-build).
#   HOST_UID / HOST_GID        Optional chown target for INSTALL_PREFIX.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${INSTALL_PREFIX:-/output}"
REPO_ROOT="${REPO_ROOT:-/repo}"
BUILD_DIR="${BUILD_DIR:-/tmp/libcudf-build}"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/setup_java_env.sh"

if [[ -z ${RAPIDS_CUDA_VERSION:-} ]]; then
  echo "Error: RAPIDS_CUDA_VERSION must be set" >&2
  exit 1
fi

CMAKE_ARGS=(
  -S "${REPO_ROOT}/cpp"
  -B "${BUILD_DIR}"
  -GNinja
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  -DCMAKE_C_COMPILER="${CC}"
  -DCMAKE_CXX_COMPILER="${CXX}"
  -DCMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER}"
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTS=OFF
  -DUSE_NVTX=ON
  -DCUDF_LARGE_STRINGS_DISABLED=ON
  -DCUDF_USE_ARROW_STATIC=ON
  -DCUDF_ENABLE_ARROW_S3=OFF
  -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON
  -DRMM_LOGGING_LEVEL=OFF
  -DCUDF_KVIKIO_REMOTE_IO=OFF
)
if [[ -n ${CMAKE_CUDA_ARCHITECTURES:-} ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi
# Forward the sccache launchers exported by setup_java_env.sh when present.
if [[ -n ${CMAKE_C_COMPILER_LAUNCHER:-} ]]; then
  CMAKE_ARGS+=("-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}")
fi
if [[ -n ${CMAKE_CXX_COMPILER_LAUNCHER:-} ]]; then
  CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
fi
if [[ -n ${CMAKE_CUDA_COMPILER_LAUNCHER:-} ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}")
fi
# setup_java_env.sh builds the static Boost archives into BOOST_PREFIX.
if [[ -d ${BOOST_PREFIX} ]]; then
  CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${BOOST_PREFIX}")
fi

rapids-logger "Configuring/building static libcudf (cuda=${RAPIDS_CUDA_VERSION})"
cudf_java_scl cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "${PARALLEL_LEVEL}"
cmake --install "${BUILD_DIR}"

# Hand the install tree back to the host user (host wrapper passes HOST_UID/GID).
if [[ -n ${HOST_UID:-} && -n ${HOST_GID:-} ]]; then
  chown -R "${HOST_UID}:${HOST_GID}" "${INSTALL_PREFIX}"
fi
if command -v sccache >/dev/null 2>&1; then
  sccache --show-adv-stats || true
fi
