#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-datetime-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION
rapids-generate-version > ./python/cudf/cudf/VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

rapids-logger "Prepending channels ${CPP_CHANNEL} and ${PYTHON_CHANNEL} to RATTLER_CHANNELS"

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "--channel" "${PYTHON_CHANNEL}" "${RATTLER_CHANNELS[@]}")

rapids-logger "Building dask-cudf"

rapids-telemetry-record build-dask-cudf.log \
    rattler-build build --recipe conda/recipes/dask-cudf \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rapids-logger "Building cudf-polars"

rapids-telemetry-record build-cudf-polars.log \
    rattler-build build --recipe conda/recipes/cudf-polars \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rapids-logger "Building custreamz"

rapids-telemetry-record build-custreamz.log \
    rattler-build build --recipe conda/recipes/custreamz \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cudf cudf --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
