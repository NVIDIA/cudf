#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Propose an update of ci/cudf_polars/gpu_test_run_list.json after a nightly run
# whose audit reported only additions. The change is a pull request for a
# maintainer to review. A run that lost GPU execution, has stale nodes, or has
# incomplete evidence is never proposed: the audit job keeps failing for those.
#
# Environment:
#   GH_TOKEN            token that can write contents and pull requests
#   GITHUB_REPOSITORY   owner/name
#   GITHUB_RUN_ID       the nightly run to read
#   TESTED_SHA          the revision the nightly run tested
#   DRY_RUN             if set, print the decision and change nothing

set -euo pipefail

branch=ci/auto-update-gpu-run-list
run_list=ci/cudf_polars/gpu_test_run_list.json
work=$(mktemp -d)

# Rerunning failed jobs does not repeat the diagnostics job, so take the newest.
artifact=$(gh api --paginate \
    "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/artifacts?per_page=100" \
    --jq '.artifacts[].name | select(test("^cudf-polars-fallback-diagnostics-[0-9]+$"))' \
    | sort -t- -k4 -n | tail -n1)
if [[ -z "${artifact}" ]]; then
    echo "No diagnostics artifact in run ${GITHUB_RUN_ID}; nothing to propose."
    exit 0
fi
gh run download "${GITHUB_RUN_ID}" --repo "${GITHUB_REPOSITORY}" \
    --name "${artifact}" --dir "${work}"

drift="${work}/run-list-drift.json"
candidate="${work}/gpu-run-list-candidate.json"

status=$(jq -r .status "${drift}")
if [[ "${status}" != "drift" ]]; then
    echo "Audit status is '${status}'; nothing to propose."
    exit 0
fi
if jq -e '(.errors | length > 0)
          or ([.differences[] | (.lost_gpu_execution | length) + (.stale | length)] | add > 0)' \
    "${drift}" > /dev/null; then
    echo "The drift includes errors, lost GPU execution or stale nodes; a person must review it."
    exit 0
fi
shas=$(jq -r '[.runs[].cudf_sha] | unique | join(",")' "${work}/fallback-diagnostics.json")
if [[ "${shas}" != "${TESTED_SHA}" ]]; then
    echo "Diagnostics are for ${shas}, not ${TESTED_SHA}; nothing to propose."
    exit 0
fi

added=$(jq '[.differences[].added | length] | add' "${drift}")
summary=$(jq -r '.differences | to_entries[]
    | "- `\(.key)`: \(.value.added | length) added"' "${drift}")
echo "Proposing ${added} additions from ${artifact}."
if [[ -n "${DRY_RUN:-}" ]]; then
    echo "${summary}"
    exit 0
fi

base_oid=$(gh api "repos/${GITHUB_REPOSITORY}/git/ref/heads/main" --jq .object.sha)
open_pr=$(gh pr list --repo "${GITHUB_REPOSITORY}" --head "${branch}" --state open \
    --json number --jq '.[0].number // empty')
if [[ -n "${open_pr}" ]]; then
    head_oid=$(gh api "repos/${GITHUB_REPOSITORY}/git/ref/heads/${branch}" --jq .object.sha)
else
    # A branch without an open PR is left over from a merged or closed proposal.
    gh api -X DELETE "repos/${GITHUB_REPOSITORY}/git/refs/heads/${branch}" 2> /dev/null || true
    gh api "repos/${GITHUB_REPOSITORY}/git/refs" -f "ref=refs/heads/${branch}" \
        -f "sha=${base_oid}" > /dev/null
    head_oid=${base_oid}
fi

# A commit made through the API is signed by GitHub for the token's identity.
jq -n --arg repo "${GITHUB_REPOSITORY}" --arg branch "${branch}" --arg oid "${head_oid}" \
    --arg path "${run_list}" --rawfile contents "${candidate}" \
    --arg headline "Update the cudf-polars GPU test run list" \
    --arg body "Add the upstream Polars tests that now run on the GPU, from nightly run ${GITHUB_RUN_ID} at ${TESTED_SHA}." \
    '{query: "mutation($input: CreateCommitOnBranchInput!) { createCommitOnBranch(input: $input) { commit { oid } } }",
      variables: {input: {
        branch: {repositoryNameWithOwner: $repo, branchName: $branch},
        expectedHeadOid: $oid,
        message: {headline: $headline, body: $body},
        fileChanges: {additions: [{path: $path, contents: ($contents | @base64)}]}}}}' \
    | gh api graphql --input - > /dev/null

pr_body="${work}/pr-body.md"
cat > "${pr_body}" << BODY
The nightly \`cudf-polars-test-audit\` reported drift with only additions. This regenerates \`${run_list}\` from the audit's candidate.

${summary}

- Nightly run: https://github.com/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}
- Tested revision: \`${TESTED_SHA}\`
- No test lost GPU execution and no node is stale.

Review the added tests as the \`update-cudf-polars-polars-tests-selection\` skill describes, then merge. This pull request is updated by each nightly that finds new additions.
BODY
if [[ -z "${open_pr}" ]]; then
    gh pr create --repo "${GITHUB_REPOSITORY}" --base main --head "${branch}" \
        --title "Update the cudf-polars GPU test run list" \
        --body-file "${pr_body}" --label improvement --label non-breaking
else
    gh pr edit "${open_pr}" --repo "${GITHUB_REPOSITORY}" --body-file "${pr_body}"
fi
