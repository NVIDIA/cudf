#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Require the shared infrastructure so jobs do not silently bootstrap old images.
if ! command -v rapids-init-uv >/dev/null 2>&1 || ! command -v rapids-uv-pip >/dev/null 2>&1; then
    echo "cuDF wheel CI requires updated images with rapids-init-uv, rapids-uv-pip, and uv>=0.12.0" >&2
    return 1
fi
source rapids-init-uv
