#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-datetime-string

# shellcheck source=ci/build_python_common.sh
source ./ci/build_python_common.sh

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION
rapids-generate-version > ./python/cudf/cudf/VERSION

rapids-logger "Begin py build"

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

PARALLEL_OUTPUT_DIR="${RAPIDS_CONDA_BLD_OUTPUT_DIR}-parallel"
builds=()
for package in dask-cudf cudf-polars custreamz; do
  run_logged_build "${package}-parallel-build.log" \
    build_conda_package "${package}" "${PARALLEL_OUTPUT_DIR}/${package}" &
  builds+=("$!" "${package}-parallel-build.log")
done
wait_for_builds "${builds[@]}"
collect_conda_packages "${PARALLEL_OUTPUT_DIR}"/*

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cudf cudf --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
