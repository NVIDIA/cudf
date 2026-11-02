#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CI images can contain an older uv without support for transitive prereleases.
CUDF_UV_VERSION=0.12.0
CUDF_UV_BIN_DIR="${GITHUB_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.cache/uv-bin"
CUDF_INSTALLED_UV_VERSION=$("${CUDF_UV_BIN_DIR}/uv" --version 2>/dev/null || true)
if [[ "${CUDF_INSTALLED_UV_VERSION}" != "uv ${CUDF_UV_VERSION}" && "${CUDF_INSTALLED_UV_VERSION}" != "uv ${CUDF_UV_VERSION} "* ]]; then
    CUDF_UV_INSTALLER=$(mktemp)
    curl --fail --silent --show-error --location --retry 10 \
        "https://astral.sh/uv/${CUDF_UV_VERSION}/install.sh" -o "${CUDF_UV_INSTALLER}" || return
    UV_INSTALL_DIR="${CUDF_UV_BIN_DIR}" UV_NO_MODIFY_PATH=1 sh "${CUDF_UV_INSTALLER}" || return
    rm -f "${CUDF_UV_INSTALLER}"
fi
export PATH="${CUDF_UV_BIN_DIR}:${PATH}"
