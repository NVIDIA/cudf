#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helper for the sharded pandas-tests jobs. Sourced by summary.sh (PR
# side) and merge-nightly.sh (nightly side): both download every shard's partial
# per-module summary from the current run and merge them into one file. The two
# callers differ only in how they react to a missing shard, so that decision is
# left to them and this returns non-zero instead of exiting.

# merge_shard_results <artifact_prefix> <results_filename> <num_shards> <output>
merge_shard_results() {
    local prefix=$1 filename=$2 num_shards=$3 output=$4
    local shard
    local results=()

    for ((shard = 0; shard < num_shards; shard++)); do
        if ! gh run download "${GITHUB_RUN_ID}" \
            --repo "${GITHUB_REPOSITORY}" \
            --name "${prefix}-${shard}" \
            --dir "shard-${shard}"; then
            rapids-logger "Could not download results for shard ${shard}."
            return 1
        fi
        results+=("shard-${shard}/${filename}")
    done

    rapids-logger "Merging ${#results[@]} shard result file(s)"
    python ci/cudf_pandas_scripts/pandas-tests/merge-results.py \
        "${results[@]}" > "${output}"
}
