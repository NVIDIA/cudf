#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Aggregate the results of the sharded pandas-tests PR jobs.
#
# Each shard uploads its partial per-module summary as the GitHub artifact
# "pandas-test-pr-results-<shard_id>". This job downloads them all, merges them
# into a single pr-results.json, and posts the diff against the latest nightly
# ("main") results to the job summary.
#
# Usage:
#   summary.sh <num_shards>
#
# This step is purely informational, so it never fails the workflow; the
# pass/fail signal for the suite comes from the individual shard jobs.

# No `set -e`: this step is best effort and must never fail the workflow, so
# every fallible command is guarded explicitly and the script always exits 0.
set -uo pipefail

source rapids-init-pip
# shellcheck source=ci/cudf_pandas_scripts/pandas-tests/shard-results.sh
source ci/cudf_pandas_scripts/pandas-tests/shard-results.sh

NUM_SHARDS=${1:?usage: summary.sh <num_shards>}
RAPIDS_FULL_VERSION=$(<./VERSION)

rapids-logger "Aggregating pandas-tests results from ${NUM_SHARDS} shards"

# job-summary.py renders markdown tables with pandas; tabulate backs to_markdown.
if ! rapids-pip-retry install pandas tabulate; then
    rapids-logger "Could not install summary dependencies; skipping summary."
    exit 0
fi

# Download and merge every shard's partial results. A shard that fails never
# reaches its upload step, so its results are simply absent; summarizing anyway
# would diff part of this PR's suite against the whole nightly baseline and
# report every test in the missing shard as removed, which is worse than
# printing nothing at all.
if ! merge_shard_results "pandas-test-pr-results" "pr-results.json" \
    "${NUM_SHARDS}" pr-results.json; then
    rapids-logger "Could not assemble all ${NUM_SHARDS} shards; skipping the summary rather than reporting a partial diff."
    exit 0
fi

# Fetch the latest successful nightly results to diff against.
MAIN_RUN_ID=$(
    gh run list                       \
        -w "Pandas Test Job"          \
        -b "$(<./RAPIDS_BRANCH)"      \
        --repo 'NVIDIA/cudf'        \
        --status success              \
        --limit 7                     \
        --json 'createdAt,databaseId' \
        --jq 'sort_by(.createdAt) | reverse | .[0].databaseId // empty' || true
)

if [[ -z "${MAIN_RUN_ID}" ]]; then
    rapids-logger "No nightly main results found; skipping diff."
    exit 0
fi

rapids-logger "Fetching latest available results from nightly: ${MAIN_RUN_ID}"
if ! gh run download                  \
    --repo 'NVIDIA/cudf'        \
    --name main-results.json \
    "${MAIN_RUN_ID}"; then
    rapids-logger "Could not download nightly results; skipping diff."
    exit 0
fi

# Compute the diff and prepare the job summary (best effort).
if ! python ci/cudf_pandas_scripts/pandas-tests/job-summary.py \
    main-results.json pr-results.json "${RAPIDS_FULL_VERSION}" >> "$GITHUB_STEP_SUMMARY"; then
    rapids-logger "Failed to render the job summary."
fi

exit 0
