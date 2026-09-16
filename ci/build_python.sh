#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-datetime-string

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

sccache --stop-server 2>/dev/null || true

rapids-logger "Building pylibcudf"

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler-build.prefix.dev/latest/tips_and_tricks/
rapids-telemetry-record build-pylibcudf.log \
  rattler-build build --recipe conda/recipes/pylibcudf \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}" 2>&1 | tee pylibcudf-build-output.log

rapids-logger "Checking for Cython performance warnings in pylibcudf"
if grep -Fq "performance hint:" pylibcudf-build-output.log; then
  echo "Cython performance hints found in pylibcudf build:"
  grep -F "performance hint:" pylibcudf-build-output.log
  exit 1
fi

rapids-telemetry-record sccache-stats-pylibcudf.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Building cudf"

rapids-telemetry-record build-cudf.log \
   rattler-build build --recipe conda/recipes/cudf \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats-cudf.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Building cudf_kafka"

rapids-telemetry-record build-cudf_kafka.log \
    rattler-build build --recipe conda/recipes/cudf_kafka \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats-cudf_kafka.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Building cudf_streaming"

rapids-telemetry-record build-cudf_streaming.log \
    rattler-build build --recipe conda/recipes/cudf_streaming \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}" 2>&1 | tee cudf_streaming-build-output.log

rapids-logger "Checking for Cython performance warnings in cudf_streaming"
if grep -Fq "performance hint:" cudf_streaming-build-output.log; then
  echo "Cython performance hints found in cudf_streaming build:"
  grep -F "performance hint:" cudf_streaming-build-output.log
  exit 1
fi

rapids-telemetry-record sccache-stats-cudf_streaming.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
