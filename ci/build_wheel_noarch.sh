#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

# Build pure-Python wheels independently of the non-noarch wheel chain.

SCCACHE_SERVER_PORT=4227 run_logged_build dask-cudf-parallel-build.log \
  build_noarch_wheel dask_cudf dask-cudf python/dask_cudf 10M &
dask_pid=$!
SCCACHE_SERVER_PORT=4228 run_logged_build cudf-polars-parallel-build.log \
  build_noarch_wheel cudf_polars cudf-polars python/cudf_polars 10M &
polars_pid=$!
wait_for_builds "${dask_pid}" dask-cudf-parallel-build.log "${polars_pid}" cudf-polars-parallel-build.log
