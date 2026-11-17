#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the ctests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf_streaming/"
# Fall back to the build directory (devcontainer environments)
script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
devcontainers_test_location="${script_dir}/../cpp/libcudf_streaming/build/latest"
buildsh_test_location="${STREAMING_LIB_BUILD_DIR:-${script_dir}/../cpp/libcudf_streaming/build}"

if [[ -d "${installed_test_location}" ]]; then
    cd "${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    cd "${devcontainers_test_location}"
elif [[ -d "${buildsh_test_location}" ]]; then
    cd "${buildsh_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    echo "  - ${buildsh_test_location}" >&2
    exit 1
fi

# OpenMPI specific options (CI runs as root)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1

ctest --output-on-failure --no-tests=error "$@"
