#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Local GPU wrapper around ci/test_packaged_java.sh.
#
# Resolves the classifier JAR from a test_java_build_local.sh --work-dir and
# runs the packaged-JAR tests in a GPU-enabled ci-wheel container. Set
# RAPIDS_CUDA_VERSION to select which classifier to test (default: 12.9.2).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/ci_wheel_image.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/java_classifier.sh"

print_help() {
  cat << EOF
Usage: test_packaged_java_local.sh --work-dir <path>

Uses the output from test_java_build_local.sh. Set RAPIDS_CUDA_VERSION to
select the classifier to test (default: 12.9.2).
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  print_help
  exit 0
fi
if [[ $# -ne 2 ]]; then
  print_help
  exit 1
fi
if [[ $1 != "-w" && $1 != "--work-dir" ]]; then
  print_help
  exit 1
fi

WORK_DIR="$(cd "$2" && pwd)"
CUDA_VERSION="${RAPIDS_CUDA_VERSION:-12.9.2}"
CLASSIFIER="$(cudf_java_maven_classifier "${CUDA_VERSION}")"
JAR_PATH="$(realpath "$(cudf_java_find_classifier_jar "${WORK_DIR}/jars/${CLASSIFIER}" "${CLASSIFIER}")")"
IMAGE="$(cudf_java_ci_wheel_image "${CUDA_VERSION}")"

echo "Running cuDF Java tests"
echo "  image:        ${IMAGE}"
echo "  cuda version: ${CUDA_VERSION}"
echo "  classifier:   ${CLASSIFIER}"
echo "  jar:          ${JAR_PATH}"

docker run --rm --gpus all \
  --volume "${REPO_ROOT}:/repo" \
  --volume "${JAR_PATH}:/product/cudf.jar:ro" \
  --workdir /repo \
  --env RAPIDS_CUDA_VERSION="${CUDA_VERSION}" \
  --env JAVA_JAR=/product/cudf.jar \
  --env LIBCUDF_LARGE_STRINGS_ENABLED=0 \
  "${IMAGE}" bash /repo/ci/test_packaged_java.sh
