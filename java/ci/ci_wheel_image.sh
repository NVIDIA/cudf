#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image-tag helper for Java packaging containers. Source, do not execute.
#
# cudf_java_ci_wheel_image builds the rapidsai/ci-wheel tag from VERSION +
# CUDA_VERSION + RAPIDS_PY_VERSION. CUDA_VERSION must be the full toolkit
# version used in the image tag (e.g. 12.9.2), matching RAPIDS_CUDA_VERSION
# in CI.

_java_ci_image_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_java_ci_image_repo_root="${REPO_ROOT:-$(git -C "${_java_ci_image_script_dir}" rev-parse --show-toplevel)}"

cudf_java_ci_wheel_image() {
  local cuda_version=${1:?CUDA version required}
  local rapids_ver py_ver
  rapids_ver="$(head -1 "${_java_ci_image_repo_root}/VERSION" | cut -d. -f1,2)"
  py_ver="${RAPIDS_PY_VERSION:-3.11}"
  echo "rapidsai/ci-wheel:${rapids_ver}-cuda${cuda_version}-rockylinux8-py${py_ver}"
}
