#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Self-contained packaging of the cuDF Java JAR for a single classifier.
#
# Consumes a prebuilt static libcudf install tree (from build_static_libcudf.sh),
# compiles the JNI layer against it inside a throwaway RAPIDS ci-conda container,
# and emits the single classifier JAR (plus its POM) to a per-classifier
# subdirectory under --output-dir. This script is layout-agnostic: it produces
# one classifier's artifacts and knows nothing about the combined
# Maven-repository layout (see java/ci/assemble_maven_repo.sh).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/argparse.sh"

LIBCUDF_DIR=""
OUTPUT_DIR=""
CUDA_VERSION=""
CMAKE_CUDA_ARCHITECTURES=""
PARALLEL_LEVEL="$(nproc)"

print_help() {
  cat << EOF

Usage: build_cudf_java_jar.sh --libcudf-dir <path> --output-dir <path> \\
                              --cuda-version <ver> [OPTIONS]

Packages the cuDF Java JAR for a single classifier inside a RAPIDS ci-conda
container, linking against a prebuilt static libcudf. Always builds for the
host architecture (uname -m). The build image is fixed to
rapidsai/ci-conda:<rapids_version>-latest (version derived from the VERSION
file).

The classifier is derived from --cuda-version (major) + host arch (uname -m),
mirroring the pom.xml Groovy logic: "cuda<major>" for x86_64,
"cuda<major>-arm64" for aarch64. The classifier JAR + POM are written to
<output-dir>/<classifier>/. Concurrent invocations targeting different
classifiers are safe.

REQUIRED:
    -l, --libcudf-dir    Static libcudf install tree produced by
                         build_static_libcudf.sh.
    -o, --output-dir     Host parent directory. The script creates and writes
                         to <output-dir>/<classifier>/, which must not already
                         exist.
    -c, --cuda-version   CUDA version to build for (e.g. "12.9" or "12.9.1").
                         Must match --cuda-version of the static libcudf tree;
                         determines the cuda12/cuda13 classifier.

OPTIONS:
    -A, --cmake-cuda-architectures
                         Override the CUDA architecture list (e.g. "80" or
                         "80;90"). When unset, uses cuDF's default RAPIDS
                         architecture list. Must match the value passed to
                         build_static_libcudf.sh when producing the static
                         libcudf tree in --libcudf-dir, or device linking of
                         libcudfjni.so against libcudf.a will fail.
    -j, --parallel       Build parallelism (default: nproc = ${PARALLEL_LEVEL}).
    -h, --help           Show this help message.

EXAMPLES:
    build_cudf_java_jar.sh -l /tmp/libcudf-cuda12 -o /tmp/jars -c 12.9
    build_cudf_java_jar.sh -l /tmp/libcudf-cuda13 -o /tmp/jars -c 13.3 -A 80
    # writes:
    #   /tmp/jars/cuda12/cudf-<version>-cuda12.jar
    #   /tmp/jars/cuda12/cudf-<version>.pom

EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        print_help
        exit 0
        ;;
      -l|--libcudf-dir)
        require_value "$1" "$2"
        LIBCUDF_DIR=$2
        shift 2
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

require_arg --libcudf-dir  "${LIBCUDF_DIR}"
require_arg --output-dir   "${OUTPUT_DIR}"
require_arg --cuda-version "${CUDA_VERSION}"

if [[ ! -d ${LIBCUDF_DIR} ]]; then
  echo "Error: --libcudf-dir '${LIBCUDF_DIR}' does not exist."
  exit 1
fi

# Derive the Maven classifier from --cuda-version major + host arch, mirroring
# the pom.xml Groovy logic: "cuda<major>" for x86_64, "cuda<major>-arm64" for
# aarch64.
CUDA_MAJOR="$(echo "${CUDA_VERSION}" | cut -d. -f1)"
HOST_ARCH="$(uname -m)"
case "${HOST_ARCH}" in
  x86_64)
    CLASSIFIER="cuda${CUDA_MAJOR}"
    ;;
  aarch64|arm64)
    CLASSIFIER="cuda${CUDA_MAJOR}-arm64"
    ;;
  *)
    echo "Error: Unsupported host arch '${HOST_ARCH}' (expected x86_64 or aarch64)" >&2
    exit 1
    ;;
esac

RAPIDS_VERSION="$(head -1 "${REPO_ROOT}/VERSION" | cut -d. -f1,2)"
IMAGE="rapidsai/ci-conda:${RAPIDS_VERSION}-latest"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
LIBCUDF_DIR="$(cd "${LIBCUDF_DIR}" && pwd)"

CLASSIFIER_OUT="${OUTPUT_DIR}/${CLASSIFIER}"
if [[ -e ${CLASSIFIER_OUT} ]]; then
  echo "Error: classifier output '${CLASSIFIER_OUT}' already exists. Remove it before re-running." >&2
  exit 1
fi
mkdir -p "${CLASSIFIER_OUT}"

# Per-classifier scratch dir for Maven's java/target/. Nested bind-mount over
# /repo/java/target inside the container isolates concurrent invocations
# (each classifier gets its own target/). The `.mvn-temp-target/` prefix
# keeps this dir invisible to the default `*/` globbing in
# assemble_maven_repo.sh's classifier discovery loop.
#
# Recreate the scratch on every launch: the in-container mvn cannot clean a
# bind-mount point (rmdir on /repo/java/target fails with EBUSY), so the
# host wrapper is responsible for guaranteeing a clean starting target/.
TARGET_SCRATCH="${OUTPUT_DIR}/.mvn-temp-target/${CLASSIFIER}"
rm -rf "${TARGET_SCRATCH}"
mkdir -p "${TARGET_SCRATCH}"

echo "Packaging cuDF Java JAR"
echo "  image:        ${IMAGE}"
echo "  cuda version: ${CUDA_VERSION}"
echo "  classifier:   ${CLASSIFIER}"
echo "  parallel:     ${PARALLEL_LEVEL}"
echo "  libcudf dir:  ${LIBCUDF_DIR}"
echo "  output dir:   ${CLASSIFIER_OUT}"
echo "  target dir:   ${TARGET_SCRATCH}"
if [[ -n ${CMAKE_CUDA_ARCHITECTURES} ]]; then
  echo "  cmake cuda archs: ${CMAKE_CUDA_ARCHITECTURES}"
fi

DOCKER_ARGS=(
  --rm
  --volume "${REPO_ROOT}:/repo"
  --volume "${LIBCUDF_DIR}:/libcudf:ro"
  --volume "${CLASSIFIER_OUT}:/output"
  --volume "${TARGET_SCRATCH}:/repo/java/target"
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
  bash /repo/java/ci/build_cudf_java_jar_in_container.sh

# Post-run: assert exactly one main classifier JAR + one POM, and that the
# JAR's classifier suffix matches the subdir name we chose (catches pom drift).
PRODUCED_JAR=""
for candidate in "${CLASSIFIER_OUT}"/cudf-*-"${CLASSIFIER}".jar; do
  if [[ -f "${candidate}" ]]; then
    if [[ -n "${PRODUCED_JAR}" ]]; then
      echo "Error: multiple JARs matching cudf-*-${CLASSIFIER}.jar found in ${CLASSIFIER_OUT}"
      ls -1 "${CLASSIFIER_OUT}"
      exit 1
    fi
    PRODUCED_JAR=${candidate}
  fi
done

if [[ -z "${PRODUCED_JAR}" ]]; then
  echo "Error: no cudf-*-${CLASSIFIER}.jar found in ${CLASSIFIER_OUT}"
  ls -1 "${CLASSIFIER_OUT}" || true
  exit 1
fi

PRODUCED_POM=""
for candidate in "${CLASSIFIER_OUT}"/cudf-*.pom; do
  if [[ -f "${candidate}" ]]; then
    if [[ -n "${PRODUCED_POM}" ]]; then
      echo "Error: multiple POMs found in ${CLASSIFIER_OUT}"
      ls -1 "${CLASSIFIER_OUT}"
      exit 1
    fi
    PRODUCED_POM=${candidate}
  fi
done

if [[ -z "${PRODUCED_POM}" ]]; then
  echo "Error: no cudf-*.pom found in ${CLASSIFIER_OUT}"
  ls -1 "${CLASSIFIER_OUT}" || true
  exit 1
fi

echo "cuDF Java JAR build succeeded:"
echo "  $(basename "${PRODUCED_JAR}")"
echo "  $(basename "${PRODUCED_POM}")"
