#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./ci/test_cpp_common.sh

diagnostics_dir="${RAPIDS_ARTIFACTS_DIR}/transform-diagnostics"
mkdir -p "${diagnostics_dir}"
conda list --json > "${diagnostics_dir}/environment-before-gdb.json"
# Keep the solver from changing the CUDA/RMM stack under investigation.
rapids-mamba-retry install --yes --freeze-installed -n test gdb
conda list --json > "${diagnostics_dir}/environment-after-gdb.json"
nvidia-smi -q > "${diagnostics_dir}/nvidia-smi.txt"
git rev-parse HEAD > "${diagnostics_dir}/commit.txt"
gdb --version > "${diagnostics_dir}/gdb-version.txt"

python ci/utils/diagnose_transform.py \
    --test-dir "${CONDA_PREFIX}/bin/gtests/libcudf" \
    --output "${diagnostics_dir}"
