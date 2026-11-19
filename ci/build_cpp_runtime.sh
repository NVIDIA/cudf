#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export BUILD_CPP_TESTS=false
source "$(dirname "${BASH_SOURCE[0]}")/build_cpp.sh"
