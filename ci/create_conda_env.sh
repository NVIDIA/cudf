#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

env_yaml=$1
shift
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
temp_dir=$(mktemp -d)
trap 'rm -rf "${temp_dir}"' EXIT

python "${ci_dir}/utils/split_conda_pip.py" \
    "${env_yaml}" "${temp_dir}/env.yaml" "${temp_dir}/requirements.txt"
rapids-mamba-retry env create --yes -f "${temp_dir}/env.yaml" "$@"
if [[ -s "${temp_dir}/requirements.txt" ]]; then
    # shellcheck source=ci/init_uv.sh
    source "${ci_dir}/init_uv.sh"
    # The caller has not activated this environment; target it instead of the parent.
    env_python=$(conda run "$@" python -c 'import sys; print(sys.executable)')
    uv pip install --python "${env_python}" -r "${temp_dir}/requirements.txt"
fi
