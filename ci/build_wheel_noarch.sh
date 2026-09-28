#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

# Build pure-Python wheels independently of the non-noarch wheel chain.

build_noarch_wheel dask_cudf dask-cudf python/dask_cudf 10M
build_noarch_wheel cudf_polars cudf-polars python/cudf_polars 10M
