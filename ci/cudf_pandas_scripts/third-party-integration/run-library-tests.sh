#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cleanup() {
    rm "${TEST_DIR}"/results-*.pickle
}

trap cleanup EXIT

runtest() {
    local lib=$1
    local mode=$2

    echo "Running tests for $lib in $mode mode"
    local plugin=()
    if [ "$mode" = "cudf" ]; then
        plugin=("-p cudf.pandas")
    fi

    pytest \
    "${plugin[@]}" \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --numprocesses="${NUM_PROCESSES}" \
    --dist=worksteal \
    "${TEST_DIR}"/test_"${lib}"*.py
}

main() {
    local lib=$1

    # generation phase
    runtest "${lib}" "gold"
    runtest "${lib}" "cudf"

    # assertion phase
    pytest \
    --compare \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --numprocesses="${NUM_PROCESSES}" \
    --dist=worksteal \
    "${TEST_DIR}"/test_"${lib}"*.py
}

main "$@"
