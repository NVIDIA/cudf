#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cleanup() {
    rm -f "${TEST_DIR}"/results-*.pickle
}

runtest() {
    local mode=$1
    shift

    echo "Running tests in $mode mode"
    local workers=${NUM_PROCESSES} args=()
    case $mode in
        cudf) args=("-p cudf.pandas") ;;
        compare)
            args=(--compare -p cudf.pandas)
            workers=${COMPARE_NUM_PROCESSES}
            ;;
    esac

    pytest \
    "${args[@]}" \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --durations=50 --durations-min=1 \
    --numprocesses="${workers}" \
    --dist=worksteal \
    "$@"
}

main() {
    local environment=$1 repo_root lib
    shift
    repo_root=$(git rev-parse --show-toplevel)
    TEST_DIR=${TEST_DIR:-${repo_root}/python/cudf/cudf_pandas_tests/third_party_integration_tests/tests}
    trap cleanup EXIT

    set +u
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "$environment"
    set -u

    NUM_PROCESSES=${NUM_PROCESSES:-4}
    COMPARE_NUM_PROCESSES=${COMPARE_NUM_PROCESSES:-0}
    local test_files=()
    for lib in "$@"; do
        test_files+=("${TEST_DIR}"/test_"${lib}"*.py)
        if [[ $lib == tensorflow ]]; then
            NUM_PROCESSES=0
            COMPARE_NUM_PROCESSES=0
        fi
    done
    local status=0 workers mode
    for workers in "${NUM_PROCESSES}" "${COMPARE_NUM_PROCESSES}"; do
        if [[ ! $workers =~ ^(0|[1-9][0-9]*)$ ]]; then
            printf 'Worker counts must be nonnegative integers: %s\n' "$workers" >&2
            return 1
        fi
    done

    for mode in gold cudf compare; do
        runtest "$mode" "${test_files[@]}" || status=1
    done

    return "$status"
}

main "$@"
