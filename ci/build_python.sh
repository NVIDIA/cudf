#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-datetime-string

# shellcheck source=ci/build_python_common.sh
source ./ci/build_python_common.sh

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION
rapids-generate-version > ./python/cudf/cudf/VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

rapids-logger "Prepending channel ${CPP_CHANNEL} to RATTLER_CHANNELS"

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")

# Package-specific servers retain cache prefixes and statistics during concurrent builds.
SCCACHE_SERVER_PORT=4226 run_logged_build pylibcudf-serial-build.log \
  build_conda_package pylibcudf "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" &
wait_for_builds "$!" pylibcudf-serial-build.log
RATTLER_CHANNELS=("--channel" "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" "${RATTLER_CHANNELS[@]}")

# Stable build prefixes preserve sccache hits across CI runs.
PARALLEL_OUTPUT_DIR="${RAPIDS_CONDA_BLD_OUTPUT_DIR}-parallel"
builds=()
packages=(cudf cudf_kafka cudf_streaming)
for index in "${!packages[@]}"; do
  SCCACHE_SERVER_PORT=$((4227 + index)) \
    run_logged_build "${packages[index]}-parallel-build.log" \
      build_conda_package "${packages[index]}" "${PARALLEL_OUTPUT_DIR}/${packages[index]}" &
  builds+=("$!" "${packages[index]}-parallel-build.log")
done
wait_for_builds "${builds[@]}"
collect_conda_packages "${PARALLEL_OUTPUT_DIR}"/*

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
