#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Self-contained build of a static libcudf install tree.
#
# Pulls the RAPIDS ci-conda image, builds libcudf with BUILD_SHARED_LIBS=OFF
# inside a throwaway container, and installs the static libcudf tree (libcudf.a
# plus its static dependencies) into a directory on the host. No GPU is required
# to build.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/argparse.sh"

OUTPUT_DIR=""
CUDA_VERSION=""
CMAKE_CUDA_ARCHITECTURES=""
PARALLEL_LEVEL="$(nproc)"

print_help() {
  cat << EOF

Usage: build_static_libcudf.sh --output-dir <path> --cuda-version <ver> [OPTIONS]

Builds a static libcudf install tree inside a RAPIDS ci-conda container and
writes it to a directory on the host. Always builds for the host architecture
(uname -m). The build image is fixed to rapidsai/ci-conda:<rapids_version>-latest
(version derived from the VERSION file).

REQUIRED:
    -o, --output-dir     Host directory to receive the static install tree
                         (libcudf.a and its static dependencies).
    -c, --cuda-version   CUDA version to build for (e.g. "12.9" or "12.9.1").

OPTIONS:
    -A, --cmake-cuda-architectures
                         Override the CUDA architecture list (e.g. "80" or
                         "80;90"). When unset, uses cuDF's default RAPIDS
                         architecture list. When packaging the cuDF Java JAR
                         against this static libcudf tree, pass the same value
                         to build_cudf_java_jar.sh --cmake-cuda-architectures
                         or device linking of libcudfjni.so against libcudf.a
                         will fail.
    -j, --parallel       Build parallelism (default: nproc = ${PARALLEL_LEVEL}).
    -h, --help           Show this help message.

EXAMPLES:
    build_static_libcudf.sh --output-dir /tmp/libcudf-cuda12 --cuda-version "12.9"
    build_static_libcudf.sh -o /tmp/libcudf-cuda13 -c 13.3 -A "80"

EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        print_help
        exit 0
        ;;
      -o|--output-dir)
        require_value "$1" "$2"
        OUTPUT_DIR=$2
        shift 2
        ;;
      -c|--cuda-version)
        require_value "$1" "$2"
        CUDA_VERSION=$2
        shift 2
        ;;
      -A|--cmake-cuda-architectures)
        require_value "$1" "$2"
        CMAKE_CUDA_ARCHITECTURES=$2
        shift 2
        ;;
      -j|--parallel)
        require_value "$1" "$2"
        PARALLEL_LEVEL=$2
        shift 2
        ;;
      *)
        echo "Error: Unknown argument $1"
        print_help
        exit 1
        ;;
    esac
  done
}

parse_args "$@"

require_arg --output-dir   "${OUTPUT_DIR}"
require_arg --cuda-version "${CUDA_VERSION}"

RAPIDS_VERSION="$(head -1 "${REPO_ROOT}/VERSION" | cut -d. -f1,2)"
IMAGE="rapidsai/ci-conda:${RAPIDS_VERSION}-latest"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

echo "Building static libcudf"
echo "  image:        ${IMAGE}"
echo "  cuda version: ${CUDA_VERSION}"
echo "  parallel:     ${PARALLEL_LEVEL}"
echo "  output dir:   ${OUTPUT_DIR}"
if [[ -n ${CMAKE_CUDA_ARCHITECTURES} ]]; then
  echo "  cmake cuda archs: ${CMAKE_CUDA_ARCHITECTURES}"
fi

DOCKER_ARGS=(
  --rm
  --volume "${REPO_ROOT}:/repo"
  --volume "${OUTPUT_DIR}:/output"
  --workdir /repo
  --env RAPIDS_CUDA_VERSION="${CUDA_VERSION}"
  --env PARALLEL_LEVEL="${PARALLEL_LEVEL}"
  --env HOST_UID="$(id -u)"
  --env HOST_GID="$(id -g)"
)

if [[ -n ${CMAKE_CUDA_ARCHITECTURES} ]]; then
  DOCKER_ARGS+=(--env CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" \
  bash /repo/java/ci/build_static_libcudf_in_container.sh

if [[ -f "${OUTPUT_DIR}/lib/libcudf.a" || -f "${OUTPUT_DIR}/lib64/libcudf.a" ]]; then
  echo "Static libcudf build succeeded: ${OUTPUT_DIR}"
else
  echo "Error: expected libcudf.a not found under ${OUTPUT_DIR}/lib or ${OUTPUT_DIR}/lib64"
  exit 1
fi
