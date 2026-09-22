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

BRIDGE_DIR="${PWD}/python/cudf_polars/quent/bridge"

rapids-logger "Check cudf-polars Quent bridge"
pushd "${BRIDGE_DIR}"
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked
python -m maturin build --locked

shopt -s nullglob
generated_stubs=(target/*/build/cudf-polars-quent-*/out/_quent/__init__.pyi)
shopt -u nullglob
if ((${#generated_stubs[@]} == 0)); then
  echo "No generated Quent stub found" >&2
  exit 1
fi
popd
