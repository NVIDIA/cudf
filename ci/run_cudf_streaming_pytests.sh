#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_cudf_streaming_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_streaming/cudf_streaming/tests

# OpenMPI specific options (CI runs as root)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1

# cudf_streaming tests require MPI for the communicator fixtures, so run them under mpirun.
EXTRA_ARGS=("$@")
run_mpirun_test() {
    local nrank="$1" # Number of ranks
    echo "Running pytest with $nrank ranks"
    mpirun --oversubscribe --map-by node --bind-to none -np "$nrank" \
        python -m pytest --cache-clear --verbose "${EXTRA_ARGS[@]}" .
}

# Note, we run with many different number of ranks, which we can do as long as
# the test suite only takes seconds to run.
for nrank in 1 2 5; do
    run_mpirun_test "$nrank"
done
