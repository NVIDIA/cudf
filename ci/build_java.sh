#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CI entrypoint: static libcudf + classifier JAR for one matrix cell.
#
# Builds the static libcudf install tree and packages the matching classifier
# JAR, writing ./output_jars/<classifier>/ for the custom-job upload step.
#
# Inputs (environment variables):
#   RAPIDS_CUDA_VERSION        CUDA version, e.g. 12.9.2 (required).
#   PARALLEL_LEVEL             Build parallelism (default: nproc).
#   CMAKE_CUDA_ARCHITECTURES   Optional override for -DCMAKE_CUDA_ARCHITECTURES.
#   JAVA_WORK_DIR              Optional scratch dir (default: <repo>/.java-work).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
. "${REPO_ROOT}/java/ci/ci_wheel_image.sh"
# shellcheck disable=SC1091
. "${REPO_ROOT}/java/ci/java_classifier.sh"

if [[ -z ${RAPIDS_CUDA_VERSION:-} ]]; then
  echo "Error: RAPIDS_CUDA_VERSION must be set" >&2
  exit 1
fi

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

export RAPIDS_CUDA_VERSION
CLASSIFIER="$(cudf_java_maven_classifier "${RAPIDS_CUDA_VERSION}")"

WORK_DIR="${JAVA_WORK_DIR:-${REPO_ROOT}/.java-work}"
LIBCUDF_DIR="${WORK_DIR}/libcudf"
CLASSIFIER_OUT="${REPO_ROOT}/output_jars/${CLASSIFIER}"

cleanup_scratch() {
  # Do not delete a caller-provided JAVA_WORK_DIR.
  if [[ -z ${JAVA_WORK_DIR:-} ]]; then
    rm -rf "${WORK_DIR}"
  else
    rm -rf "${LIBCUDF_DIR}" "${BUILD_DIR}"
  fi
}
trap cleanup_scratch EXIT

rm -rf "${LIBCUDF_DIR}" "${CLASSIFIER_OUT}"
mkdir -p "${LIBCUDF_DIR}" "${CLASSIFIER_OUT}"

export REPO_ROOT
export INSTALL_PREFIX="${LIBCUDF_DIR}"
export BUILD_DIR="${WORK_DIR}/libcudf-build"
export CUDF_INSTALL_DIR="${LIBCUDF_DIR}"
export OUTPUT_DIR="${CLASSIFIER_OUT}"

rapids-logger "Building static libcudf (${CLASSIFIER})"
bash "${REPO_ROOT}/java/ci/build_static_libcudf_in_container.sh"

rapids-logger "Packaging cuDF Java JAR (${CLASSIFIER})"
bash "${REPO_ROOT}/java/ci/build_cudf_java_jar_in_container.sh"

cudf_java_assert_classifier_artifacts "${CLASSIFIER_OUT}" "${CLASSIFIER}"
ls -la "${CLASSIFIER_OUT}"
