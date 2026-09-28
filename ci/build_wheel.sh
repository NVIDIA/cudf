#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

# Build all non-noarch wheels in one local dependency chain.

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
AUDITWHEEL_EXCLUDES=(
  --exclude libcudf.so
  --exclude libcudf_streaming.so
  --exclude libkvikio.so
  --exclude libnvcomp.so.5
  --exclude librapids_logger.so
  --exclude librapidsmpf.so
  --exclude librmm.so
  --exclude libucp.so.0
  --exclude libucxx.so
  --exclude "libnvrtc.so.${RAPIDS_CUDA_MAJOR}"
  --exclude "libnvJitLink.so.${RAPIDS_CUDA_MAJOR}"
)

repair_wheel() {
  python -m auditwheel repair \
    "${AUDITWHEEL_EXCLUDES[@]}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    "$@"
}

add_wheel_constraint() {
  local package_name=$1
  local wheel_glob=$2
  local -a wheel_paths=()

  # auditwheel determines the final platform/ABI tag, so resolve its output
  # filename before recording the local direct-reference constraint.
  mapfile -t wheel_paths < <(compgen -G "${wheel_glob}")
  if (( ${#wheel_paths[@]} != 1 )); then
    echo "Expected exactly one wheel matching ${wheel_glob}, found ${#wheel_paths[@]}" >&2
    exit 1
  fi

  echo "${package_name}-${RAPIDS_PY_CUDA_SUFFIX} @ file://${wheel_paths[0]}" >> "${PIP_CONSTRAINT}"
}

# libcudf
SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON" \
  build_package_wheel libcudf libcudf python/libcudf

repair_wheel python/libcudf/dist/*

WHEEL_EXPORT_DIR="$(mktemp -d)"
unzip -d "${WHEEL_EXPORT_DIR}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*
LIBCUDF_LIBRARY="$(find "${WHEEL_EXPORT_DIR}" -type f -name libcudf.so)"
./ci/check_symbols.sh "${LIBCUDF_LIBRARY}"

if [[ "${RAPIDS_CUDA_MAJOR}" == "12" ]]; then
  libcudf_max_wheel_size=700M
else
  libcudf_max_wheel_size=350M
fi
finalize_package_wheel \
  libcudf \
  python/libcudf \
  "${libcudf_max_wheel_size}" \
  "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "${RAPIDS_CUDA_VERSION}")"

# libcudf-streaming uses the libcudf wheel built above.
add_wheel_constraint libcudf "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/libcudf_*.whl"

build_package_wheel libcudf_streaming libcudf_streaming python/libcudf_streaming

repair_wheel python/libcudf_streaming/dist/*

finalize_package_wheel \
  libcudf_streaming \
  python/libcudf_streaming \
  100M \
  "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "${RAPIDS_CUDA_VERSION}")"

add_wheel_constraint libcudf-streaming "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/libcudf_streaming_*.whl"

# All wheels in this stage use the stable Python ABI.
export RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"

# pylibcudf
build_package_wheel \
  pylibcudf \
  pylibcudf \
  python/pylibcudf \
  --log pylibcudf-wheel-build-output.log \
  --stable
check_cython_performance_hints pylibcudf pylibcudf-wheel-build-output.log

repair_wheel python/pylibcudf/dist/*

finalize_package_wheel \
  pylibcudf \
  python/pylibcudf \
  20M \
  "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"

# cudf
add_wheel_constraint pylibcudf "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/pylibcudf_*.whl"
build_package_wheel cudf cudf python/cudf --stable

repair_wheel python/cudf/dist/*

finalize_package_wheel \
  cudf \
  python/cudf \
  15M \
  "$(rapids-artifact-name wheel_python cudf cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"

# cudf-streaming
build_package_wheel \
  cudf_streaming \
  cudf-streaming \
  python/cudf_streaming \
  --log cudf-streaming-wheel-build-output.log \
  --stable
check_cython_performance_hints cudf-streaming cudf-streaming-wheel-build-output.log

repair_wheel python/cudf_streaming/dist/*

finalize_package_wheel \
  cudf_streaming \
  python/cudf_streaming \
  75M \
  "$(rapids-artifact-name wheel_python cudf-streaming cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"
