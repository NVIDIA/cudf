#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container build of a static libcudf install tree.
#
# This script runs inside the rapidsai/ci-conda container launched by
# java/ci/build_static_libcudf.sh. It generates the build_java conda toolchain
# environment, builds libcudf with BUILD_SHARED_LIBS=OFF, and installs the
# resulting static libcudf (plus its static dependencies) into /output. Then
# chowns /output to HOST_UID:HOST_GID so the host user owns the outputs.
#
# Inputs (environment variables):
#   RAPIDS_CUDA_VERSION        CUDA version, e.g. 12.9 or 12.9.1 (required).
#   PARALLEL_LEVEL             Build parallelism (default: nproc).
#   CMAKE_CUDA_ARCHITECTURES   Optional override for -DCMAKE_CUDA_ARCHITECTURES.
#   HOST_UID / HOST_GID        Chown target for /output (both required).

set -e

INSTALL_PREFIX=/output
REPO_ROOT=/repo
BUILD_DIR=/tmp/libcudf-build

. /opt/conda/etc/profile.d/conda.sh

if [[ -z ${RAPIDS_CUDA_VERSION} ]]; then
  echo "Error: RAPIDS_CUDA_VERSION must be set" >&2
  exit 1
fi

if [[ -z ${HOST_UID} || -z ${HOST_GID} ]]; then
  echo "Error: HOST_UID and HOST_GID must both be set" >&2
  exit 1
fi

if [[ -z ${PARALLEL_LEVEL} ]]; then
  PARALLEL_LEVEL=$(nproc)
fi

CUDA_MAJOR_MINOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d. -f1,2)

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generating build_java conda environment (cuda=${CUDA_MAJOR_MINOR}, arch=$(arch))"
ENV_YAML_DIR="$(mktemp -d)"
rapids-dependency-file-generator \
  --output conda \
  --file-key build_java \
  --matrix "cuda=${CUDA_MAJOR_MINOR};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n build_java
conda activate build_java

rapids-print-env

if [[ -z ${CUDACXX} ]]; then
  export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
fi
if [[ -z ${LIBCUDF_KERNEL_CACHE_PATH} ]]; then
  export LIBCUDF_KERNEL_CACHE_PATH=/tmp/rapids-kernel-cache
fi

CMAKE_ARGS=(
  -S "${REPO_ROOT}/cpp"
  -B "${BUILD_DIR}"
  -GNinja
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
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

if [[ -n ${CMAKE_CUDA_ARCHITECTURES} ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi

rapids-logger "Configuring static libcudf"
cmake "${CMAKE_ARGS[@]}"

rapids-logger "Building static libcudf with ${PARALLEL_LEVEL} jobs"
cmake --build "${BUILD_DIR}" --parallel "${PARALLEL_LEVEL}"

rapids-logger "Installing static libcudf to ${INSTALL_PREFIX}"
cmake --install "${BUILD_DIR}"

rapids-logger "Chowning ${INSTALL_PREFIX} to ${HOST_UID}:${HOST_GID}"
chown -R "${HOST_UID}:${HOST_GID}" "${INSTALL_PREFIX}"
