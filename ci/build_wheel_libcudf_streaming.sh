#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="libcudf_streaming"
package_dir="python/libcudf_streaming"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcudf wheel from this current build,
# then ensures 'libcudf_streaming' wheel builds always use the 'libcudf'
# just built in the same CI run.
LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true;use_cuda_wheels=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

./ci/build_wheel.sh "${package_name}" "${package_dir}"

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude librapidsmpf.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude libucxx.so \
    --exclude libucp.so.0 \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
