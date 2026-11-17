#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests of cudf-polars integrations with other libraries, currently PyTorch.

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

read -r -a POLARS_COMPAT_VERSIONS <<< "$(python ci/utils/get_matrix_values.py dependencies.yaml test_cudf_polars_compat polars_compat_version)"
export CUDF_EXTRA_DEPENDENCY_MATRIX="polars_compat_version=${POLARS_COMPAT_VERSIONS[-1]}"

source ./ci/test_python_common.sh test_python_other test_cudf_polars_compat test_python_pytorch

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env

rapids-logger "pytest cudf-polars PyTorch integration"
cd python/cudf_polars
python -m pytest --cache-clear -p no:benchmark \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-integrations.xml" \
  tests/streaming/test_torch_interop.py
