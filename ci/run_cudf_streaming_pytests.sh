#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_cudf_streaming_pytests.sh outside the script directory
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
cd "$(dirname "${SCRIPT_PATH}")"/../python/cudf_streaming/cudf_streaming/tests

# OpenMPI specific options (CI runs as root)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1

if [[ "${1:-}" == "--mpi-worker" ]]; then
    nrank="$2"
    junit_xml="$3"
    shift 3
    junit_args=()
    if [[ -n "${junit_xml}" ]]; then
        rank="${OMPI_COMM_WORLD_RANK:?MPI rank is unavailable}"
        junit_args=(
            "--junitxml=${junit_xml%.xml}-nrank-${nrank}-rank-${rank}.xml"
        )
    fi
    python -m pytest --cache-clear "${junit_args[@]}" "$@" .
    exit
fi

# cudf_streaming tests require MPI for the communicator fixtures, so run them under mpirun.
JUNIT_XML=""
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == --junitxml=* ]]; then
        JUNIT_XML="${arg#--junitxml=}"
    else
        EXTRA_ARGS+=("${arg}")
    fi
done

run_mpirun_test() {
    local nrank="$1" # Number of ranks
    echo "Running pytest with $nrank ranks"
    mpirun --oversubscribe --map-by node --bind-to none -np "$nrank" \
        "${SCRIPT_PATH}" --mpi-worker "${nrank}" "${JUNIT_XML}" "${EXTRA_ARGS[@]}"
}

# Note, we run with many different number of ranks, which we can do as long as
# the test suite only takes seconds to run.
for nrank in 1 2 5; do
    run_mpirun_test "$nrank"
done
