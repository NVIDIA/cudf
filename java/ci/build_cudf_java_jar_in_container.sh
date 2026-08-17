#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container packaging of the cuDF Java JAR for a single classifier.
#
# Expects a RAPIDS ci-wheel environment. Sources the toolchain from
# setup_java_env.sh, compiles the JNI layer against the prebuilt static
# libcudf at CUDF_INSTALL_DIR, and packages the cuDF Java JAR. The classifier
# JAR, sources JAR, javadoc JAR, and POM are copied to OUTPUT_DIR. When
# HOST_UID / HOST_GID are set, OUTPUT_DIR and java/target are chowned on exit
# so the host user owns the outputs.
#
# Inputs (environment variables):
#   RAPIDS_CUDA_VERSION        CUDA version, e.g. 12.9.2 (required).
#   PARALLEL_LEVEL             Build parallelism (default: nproc).
#   CMAKE_CUDA_ARCHITECTURES   Optional override for -DCMAKE_CUDA_ARCHITECTURES.
#   CUDF_INSTALL_DIR           Static libcudf install tree (default: /libcudf).
#   OUTPUT_DIR                 Artifact output dir (default: /output).
#   REPO_ROOT                  cuDF checkout (default: /repo).
#   HOST_UID / HOST_GID        Optional chown target for OUTPUT_DIR + java/target.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
REPO_ROOT="${REPO_ROOT:-/repo}"
CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR:-/libcudf}"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/setup_java_env.sh"

if [[ -z ${RAPIDS_CUDA_VERSION:-} ]]; then
  echo "Error: RAPIDS_CUDA_VERSION must be set" >&2
  exit 1
fi

_chown_outputs_on_exit() {
  if [[ -n ${HOST_UID:-} && -n ${HOST_GID:-} ]]; then
    chown -R "${HOST_UID}:${HOST_GID}" "${OUTPUT_DIR}" "${REPO_ROOT}/java/target" 2>/dev/null || true
  fi
}
trap _chown_outputs_on_exit EXIT

BUILD_ARG=(
  -B
  # Prefix every log line with HH:mm:ss.SSS to record per-plugin elapsed time.
  "-Dorg.slf4j.simpleLogger.showDateTime=true"
  "-Dorg.slf4j.simpleLogger.dateTimeFormat=HH:mm:ss.SSS"
  "-Dmaven.repo.local=${MAVEN_REPO_LOCAL:-/tmp/.m2}"
  "-Dparallel.level=${PARALLEL_LEVEL}"
  "-DskipTests=true"
  "-DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON"
  "-DCUDF_JNI_LIBCUDF_STATIC=ON"
  "-DUSE_GDS=OFF"
  # -Prelease adds the sources.jar; -Pjavadoc-jdk17 adds the javadoc.jar (built
  # against ${JDK17_HOME}/bin/javadoc). Both are required by Maven Central.
  "-Prelease"
  "-Pjavadoc-jdk17"
)
if [[ -n ${CMAKE_CUDA_ARCHITECTURES:-} ]]; then
  BUILD_ARG+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi

# Pass the toolchain + sccache launchers from setup_java_env.sh to the JNI cmake
# invocation via the pom's cmake.ccache.opts property.
CMAKE_CCACHE_OPTS=()
if [[ -n ${CMAKE_C_COMPILER_LAUNCHER:-} ]]; then
  CMAKE_CCACHE_OPTS+=("-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}")
fi
if [[ -n ${CMAKE_CXX_COMPILER_LAUNCHER:-} ]]; then
  CMAKE_CCACHE_OPTS+=("-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
fi
if [[ -n ${CMAKE_CUDA_COMPILER_LAUNCHER:-} ]]; then
  CMAKE_CCACHE_OPTS+=("-DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}")
fi
CMAKE_CCACHE_OPTS+=("-DCMAKE_C_COMPILER=${CC}" "-DCMAKE_CXX_COMPILER=${CXX}")
CMAKE_CCACHE_OPTS+=("-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")
if [[ -d ${BOOST_PREFIX} ]]; then
  CMAKE_CCACHE_OPTS+=("-DCMAKE_PREFIX_PATH=${BOOST_PREFIX}")
fi
BUILD_ARG+=("-Dcmake.ccache.opts=${CMAKE_CCACHE_OPTS[*]}")

cd "${REPO_ROOT}/java"

CUDF_VERSION="$(cudf_java_scl mvn help:evaluate -Dexpression=project.version -q -DforceStdout "${BUILD_ARG[@]}")"
rapids-logger "Packaging cuDF Java JAR ${CUDF_VERSION}"

# Omit the `clean` goal: java/target may be a bind-mount point, so `mvn clean`
# fails with EBUSY. The host wrapper recreates the scratch dir before each
# launch to guarantee target/ starts empty.
CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}" cudf_java_scl mvn package "${BUILD_ARG[@]}"

mkdir -p "${OUTPUT_DIR}"

# Order matters: *-test-sources.jar must precede *-sources.jar because
# bash's case picks the first matching pattern, and *-sources.jar would
# also match *-test-sources.jar.
MAIN_JAR=""
for candidate in target/cudf-"${CUDF_VERSION}"-*.jar; do
  case "${candidate}" in
    *-tests.jar|*-test-sources.jar)
      continue
      ;;
    *-sources.jar|*-javadoc.jar)
      cp -f "${candidate}" "${OUTPUT_DIR}/"
      ;;
    *)
      if [[ -f ${candidate} ]]; then
        if [[ -n ${MAIN_JAR} ]]; then
          echo "Error: multiple main classifier JARs under target/" >&2
          exit 1
        fi
        MAIN_JAR=${candidate}
      fi
      ;;
  esac
done

if [[ -z ${MAIN_JAR} ]]; then
  echo "Error: no classifier JAR under target/" >&2
  ls -l target/ >&2
  exit 1
fi

# Assert the release-profile artifacts landed. A missing file here means
# -Prelease or -Pjavadoc-jdk17 did not activate, or JDK17_HOME did not resolve
# to a usable javadoc binary.
for required in "${OUTPUT_DIR}/cudf-${CUDF_VERSION}-sources.jar" \
                "${OUTPUT_DIR}/cudf-${CUDF_VERSION}-javadoc.jar"; do
  if [[ ! -f ${required} ]]; then
    echo "Error: missing ${required}" >&2
    exit 1
  fi
done

cp -f "${MAIN_JAR}" "${OUTPUT_DIR}/"
cp -f pom.xml "${OUTPUT_DIR}/cudf-${CUDF_VERSION}.pom"
rapids-logger "Emitted artifacts to ${OUTPUT_DIR}"
if command -v sccache >/dev/null 2>&1; then
  sccache --show-adv-stats || true
fi
