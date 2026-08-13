#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container packaging of the cuDF Java JAR for a single classifier.
#
# This script runs inside the rapidsai/ci-conda container launched by
# java/ci/build_cudf_java_jar.sh. It generates the build_java conda toolchain
# environment, installs a JDK 17 side-prefix for the javadoc-jdk17 profile,
# compiles the JNI layer against a prebuilt static libcudf (mounted at
# /libcudf), and packages the cuDF Java JAR. The resulting classifier JAR,
# sources JAR, javadoc JAR, and POM are copied to /output. /output and
# /repo/java/target are chowned to HOST_UID:HOST_GID on exit so the host user
# owns the outputs.
#
# Inputs (environment variables):
#   RAPIDS_CUDA_VERSION        CUDA version, e.g. 12.9 or 12.9.1 (required).
#   PARALLEL_LEVEL             Build parallelism (default: nproc).
#   CMAKE_CUDA_ARCHITECTURES   Optional override for -DCMAKE_CUDA_ARCHITECTURES.
#   HOST_UID / HOST_GID        Chown target for /output and /repo/java/target
#                              (both required).

set -e

OUTPUT_DIR=/output
REPO_ROOT=/repo
CUDF_INSTALL_DIR=/libcudf

. /opt/conda/etc/profile.d/conda.sh

if [[ -z ${RAPIDS_CUDA_VERSION} ]]; then
  echo "Error: RAPIDS_CUDA_VERSION must be set" >&2
  exit 1
fi

if [[ -z ${HOST_UID} || -z ${HOST_GID} ]]; then
  echo "Error: HOST_UID and HOST_GID must both be set" >&2
  exit 1
fi

_chown_outputs_on_exit() {
  chown -R "${HOST_UID}:${HOST_GID}" "${OUTPUT_DIR}" "${REPO_ROOT}/java/target" 2>/dev/null || true
}
trap _chown_outputs_on_exit EXIT

if [[ -z ${PARALLEL_LEVEL} ]]; then
  PARALLEL_LEVEL=$(nproc)
fi

CUDA_MAJOR_MINOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d. -f1,2)

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generating build_java conda environment (cuda=${CUDA_MAJOR_MINOR}, arch=$(arch))"
ENV_YAML_DIR="$(mktemp -d)"
rapids-dependency-file-generator \
  --output conda \
  --file-key build_java \
  --matrix "cuda=${CUDA_MAJOR_MINOR};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n build_java
conda activate build_java

rapids-print-env

# The `javadoc-jdk17` profile in java/pom.xml points
# <javadocExecutable> at ${env.JDK17_HOME}/bin/javadoc. The build_java env
# above provides only JDK 8 (mvn's own JVM), so install JDK 17 into a
# dedicated prefix that JDK17_HOME can point to. The primary mvn JVM stays
# on JDK 8; only the javadoc binary is invoked from this prefix.
rapids-logger "Installing JDK 17 into /opt/jdk17 for javadoc-jdk17 profile"
rapids-mamba-retry create --yes --prefix /opt/jdk17 openjdk=17.*
export JDK17_HOME=/opt/jdk17

if [[ -z ${CUDACXX} ]]; then
  export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
fi
if [[ -z ${LIBCUDF_KERNEL_CACHE_PATH} ]]; then
  export LIBCUDF_KERNEL_CACHE_PATH=/tmp/rapids-kernel-cache
fi

BUILD_ARG=(
  -B
  # Prefix every log line with HH:mm:ss.SSS so the elapsed time of individual
  # plugin executions is recorded.
  "-Dorg.slf4j.simpleLogger.showDateTime=true"
  "-Dorg.slf4j.simpleLogger.dateTimeFormat=HH:mm:ss.SSS"
  "-Dmaven.repo.local=/tmp/.m2"
  "-Dparallel.level=${PARALLEL_LEVEL}"
  "-DskipTests=true"
  "-DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON"
  "-DCUDF_JNI_LIBCUDF_STATIC=ON"
  "-DUSE_GDS=OFF"
  # -Prelease produces the sources.jar file via maven-source-plugin;
  # -Pjavadoc-jdk17 produces the javadoc.jar file via maven-javadoc-plugin
  # running against ${env.JDK17_HOME}/bin/javadoc. Both are required by
  # Maven Central for every published release.
  "-Prelease"
  "-Pjavadoc-jdk17"
)

if [[ -n ${CMAKE_CUDA_ARCHITECTURES} ]]; then
  BUILD_ARG+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi

cd "${REPO_ROOT}/java"

CUDF_VERSION="$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout "${BUILD_ARG[@]}")"
rapids-logger "Packaging cuDF Java JAR version ${CUDF_VERSION} (libcudf: ${CUDF_INSTALL_DIR})"

# The `clean` goal is intentionally omitted: /repo/java/target is a
# bind-mount point, so when `mvn clean` attempts to remove the directory,
# it fails with EBUSY. The host wrapper (build_cudf_java_jar.sh) recreates
# the scratch dir before each container launch to guarantee target/ starts empty.
CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}" mvn package "${BUILD_ARG[@]}"

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
          echo "Error: multiple main classifier JARs matched under target/" >&2
          ls -l target/ >&2 || true
          exit 1
        fi
        MAIN_JAR=${candidate}
      fi
      ;;
  esac
done

if [[ -z ${MAIN_JAR} ]]; then
  echo "Error: no cuDF classifier JAR produced under target/" >&2
  ls -l target/ >&2 || true
  exit 1
fi

# Assert the release-profile artifacts landed. A missing file here means
# -Prelease or -Pjavadoc-jdk17 did not activate, or JDK17_HOME did not
# resolve to a usable javadoc binary.
for required in "${OUTPUT_DIR}/cudf-${CUDF_VERSION}-sources.jar" \
                "${OUTPUT_DIR}/cudf-${CUDF_VERSION}-javadoc.jar"; do
  if [[ ! -f ${required} ]]; then
    echo "Error: expected ${required} not found (mvn -Prelease -Pjavadoc-jdk17 did not produce it)" >&2
    exit 1
  fi
done

cp -f "${MAIN_JAR}" "${OUTPUT_DIR}/"
cp -f pom.xml "${OUTPUT_DIR}/cudf-${CUDF_VERSION}.pom"

rapids-logger "Emitted $(basename "${MAIN_JAR}"), cudf-${CUDF_VERSION}-sources.jar, cudf-${CUDF_VERSION}-javadoc.jar, and cudf-${CUDF_VERSION}.pom to ${OUTPUT_DIR}"
