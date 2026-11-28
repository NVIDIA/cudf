#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# A driver rejecting one display category should not hide the other counters.
for category in ECC PAGE_RETIREMENT ROW_REMAPPER; do
    printf '\n%s\n' "$category"
    nvidia-smi -q -d "$category" || true
done
