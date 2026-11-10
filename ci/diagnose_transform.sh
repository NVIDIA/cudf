#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Only the harness changes between iterations; reuse the exact original
# package build to avoid rebuilding or introducing build-environment drift.
GITHUB_RUN_ID=42678089596 source ./ci/test_cpp_common.sh

diagnostics_dir="${RAPIDS_ARTIFACTS_DIR}/transform-diagnostics"
mkdir -p "${diagnostics_dir}"
conda list --json > "${diagnostics_dir}/environment.json"
nvidia-smi -q > "${diagnostics_dir}/nvidia-smi.txt"
git rev-parse HEAD > "${diagnostics_dir}/commit.txt"

source rapids-configure-sccache
sccache --stop-server >/dev/null 2>&1 || true
sccache --dist-status > "${diagnostics_dir}/sccache-dist-status.json"
python -c 'import json, sys; assert json.load(open(sys.argv[1]))["SchedulerStatus"][1]["servers"], "Distributed sccache has no workers"' \
    "${diagnostics_dir}/sccache-dist-status.json"
sccache cc -std=c11 -shared -fPIC -g -fno-omit-frame-pointer -Wall -Wextra -Werror \
    ci/utils/crash_trace.c -o "${diagnostics_dir}/crash_trace.so"

python ci/utils/diagnose_transform.py \
    --test-dir "${CONDA_PREFIX}/bin/gtests/libcudf" \
    --crash-library "${diagnostics_dir}/crash_trace.so" \
    --output "${diagnostics_dir}"
