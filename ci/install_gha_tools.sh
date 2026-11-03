#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pin the draft gha-tools PR so every job tests the same helper implementation.
CUDF_GHA_TOOLS_REVISION=bdbf4480a9fd1d4c2c5843a0af82888b598ae9b2
if [[ "${CUDF_GHA_TOOLS_INSTALLED_REVISION:-}" != "${CUDF_GHA_TOOLS_REVISION}" ]]; then
    (
        set -euo pipefail
        tools_dir=$(dirname "$(command -v rapids-init-pip)")
        temp_dir=$(mktemp -d)
        trap 'rm -rf "${temp_dir}"' EXIT
        curl --fail --silent --show-error --location --retry 10 \
            "https://github.com/rapidsai/gha-tools/archive/${CUDF_GHA_TOOLS_REVISION}.tar.gz" \
            -o "${temp_dir}/tools.tar.gz"
        tar -xzf "${temp_dir}/tools.tar.gz" --strip-components=1 -C "${temp_dir}"
        install -m 755 "${temp_dir}/tools/"* "${tools_dir}/"
        echo "Installed gha-tools ${CUDF_GHA_TOOLS_REVISION} in ${tools_dir}"
    ) || return
    export CUDF_GHA_TOOLS_INSTALLED_REVISION="${CUDF_GHA_TOOLS_REVISION}"
fi
