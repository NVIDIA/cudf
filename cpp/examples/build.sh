#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# libcudf examples build script

set -euo pipefail

# Parallelism control
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
# Installation disabled by default
INSTALL_EXAMPLES=false

# Check for -i or --install flags to enable installation
ARGS=$(getopt -o i --long install -- "$@")
eval set -- "$ARGS"
# shellcheck disable=2078
while [ : ]; do
  case "$1" in
    -i | --install)
        INSTALL_EXAMPLES=true
        shift
        ;;
    --) shift;
        break
        ;;
  esac
done

# Root of examples
EXAMPLES_DIR=$(dirname "$(realpath "$0")")

# Set up default libcudf build directory and install prefix if conda build
if [ "${CONDA_BUILD:-"0"}" == "1" ]; then
  LIB_BUILD_DIR="${LIB_BUILD_DIR:-${SRC_DIR}/cpp/build}"
  INSTALL_PREFIX="${INSTALL_PREFIX:-${PREFIX}}"
fi

# libcudf build directory
LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${EXAMPLES_DIR}/../build")}

################################################################################
# Add individual libcudf examples build scripts down below

build_example() {
  example_dir=${1}
  example_dir="${EXAMPLES_DIR}/${example_dir}"
  build_dir="${example_dir}/build"

  # Configure
  cmake -S "${example_dir}" -B "${build_dir}" -Dcudf_ROOT="${LIB_BUILD_DIR}"
  # Build
  cmake --build "${build_dir}" -j"${PARALLEL_LEVEL}"
  # Install if needed
  if [ "$INSTALL_EXAMPLES" = true ]; then
    cmake --install "${build_dir}" --prefix "${INSTALL_PREFIX:-${example_dir}/install}"
  fi
}

example_build_pids=()

cleanup_example_builds() {
  for pid in "${example_build_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_example_builds EXIT

for example_name in \
  basic \
  hybrid_scan_io \
  strings \
  string_transforms \
  nested_types \
  parquet_inspect \
  parquet_io \
  billion_rows; do
  build_example "${example_name}" &
  example_build_pids+=("$!")
done

for pid in "${example_build_pids[@]}"; do
  wait "${pid}"
done

trap - EXIT
