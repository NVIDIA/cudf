#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source rapids-datetime-string

# shellcheck source=ci/build_python_common.sh
source ./ci/build_python_common.sh

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# The package is abi3 and has no CUDA dependency, so it is built once per architecture
# (see the matrix_filter in the calling workflows).
build_conda_package cudf-polars-quent "${RAPIDS_CONDA_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cudf-polars-quent cudf --stable)"
export RAPIDS_PACKAGE_NAME
