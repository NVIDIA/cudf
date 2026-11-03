#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Older images need a fallback until images with prerelease support are published.
CUDF_UV_VERSION=0.12.0
CUDF_AVAILABLE_UV_VERSION=$(uv --version 2>/dev/null || true)
CUDF_AVAILABLE_UV_VERSION=${CUDF_AVAILABLE_UV_VERSION#uv }
CUDF_AVAILABLE_UV_VERSION=${CUDF_AVAILABLE_UV_VERSION%% *}
if [[ -n "${CUDF_AVAILABLE_UV_VERSION}" ]] && \
    printf '%s\n' "${CUDF_UV_VERSION}" "${CUDF_AVAILABLE_UV_VERSION}" | sort --version-sort --check=quiet; then
    return 0
fi
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
