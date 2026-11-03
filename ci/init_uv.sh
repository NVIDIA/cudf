#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The draft infrastructure PRs must be testable before their releases exist.
# shellcheck source=ci/install_gha_tools.sh
source "$(dirname "${BASH_SOURCE[0]}")/install_gha_tools.sh" || return
# shellcheck source=ci/install_uv.sh
source "$(dirname "${BASH_SOURCE[0]}")/install_uv.sh" || return
source rapids-init-uv
