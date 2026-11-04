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
# shellcheck source=ci/cudf_pandas_scripts/pandas-tests/shard-results.sh
source ci/cudf_pandas_scripts/pandas-tests/shard-results.sh

NUM_SHARDS=${1:?usage: merge-nightly.sh <num_shards>}

rapids-logger "Merging pandas-tests results from ${NUM_SHARDS} shards"

# set -e propagates a missing or unmergeable shard, which is what we want here:
# see the header comment.
merge_shard_results "pandas-test-main-results" "main-results.json" \
    "${NUM_SHARDS}" main-results.json

rapids-logger "Wrote main-results.json"
