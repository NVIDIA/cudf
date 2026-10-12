#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# shellcheck source=ci/build_parallel.sh
source ./ci/build_parallel.sh

build_conda_package() (
  local package=$1
  local output_dir=$2
  local -a build_args=("${RATTLER_ARGS[@]}")
  local index
  # Separate output channels avoid concurrent repodata updates by rattler-build.
  for index in "${!build_args[@]}"; do
    if [[ "${build_args[index]}" == "--output-dir" ]]; then
      build_args[index+1]="${output_dir}"
    fi
  done

  if [[ -n "${SCCACHE_SERVER_PORT:-}" ]]; then
    sccache --stop-server >/dev/null 2>&1 || true
  fi
  rapids-logger "Building ${package}"
  rapids-telemetry-record "build-${package}.log" \
    rattler-build build --recipe "conda/recipes/${package}" \
      "${build_args[@]}" "${RATTLER_CHANNELS[@]}" 2>&1 | tee "${package}-build-output.log"

  if [[ "${package}" == "pylibcudf" || "${package}" == "cudf_streaming" ]]; then
    if grep -Fq "performance hint:" "${package}-build-output.log"; then
      echo "Cython performance hints found in ${package} build:"
      grep -F "performance hint:" "${package}-build-output.log"
      exit 1
    fi
  fi
  if [[ -n "${SCCACHE_SERVER_PORT:-}" ]]; then
    rapids-telemetry-record "sccache-stats-${package}.txt" sccache --show-adv-stats
    sccache --stop-server >/dev/null 2>&1 || true
  fi
  rm -rf "${output_dir}/build_cache"
)

collect_conda_packages() {
  local output_dir package_file subdir
  for output_dir in "$@"; do
    for package_file in "${output_dir}"/*/*.conda; do
      [[ -f "${package_file}" ]] || continue
      subdir=${package_file%/*}
      subdir=${subdir##*/}
      mkdir -p "${RAPIDS_CONDA_BLD_OUTPUT_DIR}/${subdir}"
      cp "${package_file}" "${RAPIDS_CONDA_BLD_OUTPUT_DIR}/${subdir}/"
    done
  done
  # Consumers download this directory as a channel, so its index must include all collected packages.
  conda index "${RAPIDS_CONDA_BLD_OUTPUT_DIR}"
}
