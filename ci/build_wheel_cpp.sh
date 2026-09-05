#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

# libcudf
SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON" \
  build_package_wheel libcudf libcudf python/libcudf

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
python -m auditwheel repair \
  --exclude libkvikio.so \
  --exclude libnvcomp.so.5 \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude "libnvrtc.so.${RAPIDS_CUDA_MAJOR}" \
  --exclude "libnvJitLink.so.${RAPIDS_CUDA_MAJOR}" \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/libcudf/dist/*

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
LIBCUDF_WHEELHOUSE="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"

build_package_wheel libcudf_streaming libcudf_streaming python/libcudf_streaming

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude librapidsmpf.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude libucxx.so \
  --exclude libucp.so.0 \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/libcudf_streaming/dist/*

finalize_package_wheel \
  libcudf_streaming \
  python/libcudf_streaming \
  100M \
  "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "${RAPIDS_CUDA_VERSION}")"
