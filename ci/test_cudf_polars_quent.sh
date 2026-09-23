#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

rapids-logger "Create cudf-polars Quent test environment"
. /opt/conda/etc/profile.d/conda.sh

ENV_YAML_DIR="$(mktemp -d)"
cat >"${ENV_YAML_DIR}/env.yaml" <<EOF
name: cudf_polars_quent
channels:
  - conda-forge
dependencies:
  - maturin>=1.14,<2
  - python=${RAPIDS_PY_VERSION}
  - rust=1.97
EOF

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate cudf_polars_quent
set -u

rapids-print-env

rapids-logger "Check cudf-polars Quent bridge"
exec ./ci/run_cudf_polars_quent_tests.sh
