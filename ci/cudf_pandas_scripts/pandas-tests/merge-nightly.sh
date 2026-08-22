#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Combine the results of the sharded nightly ("main") pandas-tests jobs.
#
# Each shard uploads its partial per-module summary as the GitHub artifact
# "pandas-test-main-results-<shard_id>". This job downloads them all and merges
# them into a single main-results.json, which is re-uploaded under that name so
# that PR runs keep finding it with `gh run download --name main-results.json`.
#
# Usage:
#   merge-nightly.sh <num_shards>
#
# Unlike the PR-side summary.sh, this script is NOT best effort: main-results.json
# is the baseline every PR diffs against, and a partial or missing file would show
# up as spurious "new failures" in those PRs. Failing loudly instead keeps the run
# from being picked up as the latest successful nightly.

set -euo pipefail

source rapids-init-pip

NUM_SHARDS=${1:?usage: merge-nightly.sh <num_shards>}

rapids-logger "Merging pandas-tests results from ${NUM_SHARDS} shards"

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    rapids-logger "Downloading results for shard ${shard}"
    gh run download "${GITHUB_RUN_ID}" \
        --repo "${GITHUB_REPOSITORY}" \
        --name "pandas-test-main-results-${shard}" \
        --dir "shard-${shard}"
done

SHARD_RESULTS=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    SHARD_RESULTS+=("shard-${shard}/main-results.json")
done

python ci/cudf_pandas_scripts/pandas-tests/merge-results.py \
    "${SHARD_RESULTS[@]}" > main-results.json

rapids-logger "Merged $(wc -l < main-results.json) lines into main-results.json"
