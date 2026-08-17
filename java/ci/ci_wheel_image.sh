#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image-tag helpers for Java packaging containers. Source, do not execute.
#
# cudf_java_normalize_cuda_version maps short forms (12.9, 13.3) to the full
# tags used by rapidsai/ci-wheel; cudf_java_ci_wheel_image builds the image
# name from VERSION + CUDA + RAPIDS_PY_VERSION.

_java_ci_image_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_java_ci_image_repo_root="${REPO_ROOT:-$(git -C "${_java_ci_image_script_dir}" rev-parse --show-toplevel)}"

cudf_java_normalize_cuda_version() {
  local ver=$1
  case "${ver}" in
    12.9)
      echo "12.9.2"
      ;;
    13.3)
      echo "13.3.0"
      ;;
    *)
      echo "${ver}"
      ;;
  esac
}

cudf_java_ci_wheel_image() {
  local cuda_ver=${1:?cuda version required}
  local rapids_ver cuda_full py_ver
  rapids_ver="$(head -1 "${_java_ci_image_repo_root}/VERSION" | cut -d. -f1,2)"
  cuda_full="$(cudf_java_normalize_cuda_version "${cuda_ver}")"
  py_ver="${RAPIDS_PY_VERSION:-3.11}"
  echo "rapidsai/ci-wheel:${rapids_ver}-cuda${cuda_full}-rockylinux8-py${py_ver}"
}
