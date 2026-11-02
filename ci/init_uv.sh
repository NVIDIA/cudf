#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# shellcheck source=ci/install_uv.sh
source "$(dirname "${BASH_SOURCE[0]}")/install_uv.sh" || return

# The constraint generator and local wheel references still use this filename.
PIP_CONSTRAINT="${PIP_CONSTRAINT:-$(mktemp -d)/constraints.txt}"
touch "${PIP_CONSTRAINT}"
export PIP_CONSTRAINT
export UV_CONSTRAINT="${PIP_CONSTRAINT}"

# uv ignores pip.conf and PIP_* settings. Preserve the runner's PyPI proxy and
# nightly mirror, and pip's version selection across those indexes.
export UV_INDEX_URL="${UV_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-${PIP_EXTRA_INDEX_URL:-${NV_GHA_RUNNERS_PYPI_ANACONDA_CACHE_URL:-https://pypi.anaconda.org}/rapidsai-wheels-nightly/simple}}"
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_HTTP_RETRIES=10
export UV_CACHE_DIR="${UV_CACHE_DIR:-${GITHUB_WORKSPACE:-${PWD}}/.cache/uv}"
# Build environments live on /tmp, which can be a different filesystem to the cache.
export UV_LINK_MODE=copy
