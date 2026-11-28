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
#   GITHUB_RUN_ATTEMPT  the attempt of the nightly run
#   AUDIT_ARTIFACT_PREFIX  name of the diagnostics artifact without its attempt suffix
#   DRY_RUN             if set, print the decision and change nothing
#   PROPOSAL_BRANCH     branch for the proposal (default ci/auto-update-gpu-run-list)

set -euo pipefail

branch=${PROPOSAL_BRANCH:-ci/auto-update-gpu-run-list}
run_list=ci/cudf_polars/gpu_test_run_list.json
work=$(mktemp -d)

# A rerun of only the failed jobs does not repeat the diagnostics job, so the
# current attempt may have none: use the newest attempt that does.
artifact=
for ((attempt = GITHUB_RUN_ATTEMPT; attempt >= 1; attempt--)); do
    if gh run download "${GITHUB_RUN_ID}" --repo "${GITHUB_REPOSITORY}" \
        --name "${AUDIT_ARTIFACT_PREFIX}-${attempt}" --dir "${work}" 2> /dev/null; then
        artifact=${AUDIT_ARTIFACT_PREFIX}-${attempt}
        break
    fi
done
if [[ -z "${artifact}" ]]; then
    echo "No ${AUDIT_ARTIFACT_PREFIX} artifact in run ${GITHUB_RUN_ID}; nothing to propose."
    exit 0
fi

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
# The candidate was generated from the run list at TESTED_SHA. Publishing it over a
# main that changed the run list since would drop the nodes main added.
if ! gh api "repos/${GITHUB_REPOSITORY}/contents/${run_list}?ref=${base_oid}" \
    -H "Accept: application/vnd.github.raw" | cmp -s - "${run_list}"; then
    echo "main changed ${run_list} since ${TESTED_SHA}; nothing to propose."
    exit 0
fi
open_pr=$(gh pr list --repo "${GITHUB_REPOSITORY}" --head "${branch}" --state open \
    --json number --jq '.[0].number // empty')
if [[ -n "${open_pr}" ]]; then
    head_oid=$(gh api "repos/${GITHUB_REPOSITORY}/git/ref/heads/${branch}" --jq .object.sha)
else
    # A branch without an open PR is left over from a merged or closed proposal:
    # move it to main, or create it. Nothing is deleted.
    gh api -X PATCH "repos/${GITHUB_REPOSITORY}/git/refs/heads/${branch}" \
        -f "sha=${base_oid}" -F force=true > /dev/null 2>&1 \
        || gh api "repos/${GITHUB_REPOSITORY}/git/refs" -f "ref=refs/heads/${branch}" \
            -f "sha=${base_oid}" > /dev/null
    head_oid=${base_oid}
fi

# A second nightly with the same drift must not stack an identical commit.
unchanged=false
if [[ -n "${open_pr}" ]] \
    && gh api "repos/${GITHUB_REPOSITORY}/contents/${run_list}?ref=${branch}" \
        -H "Accept: application/vnd.github.raw" 2> /dev/null | cmp -s - "${candidate}"; then
    unchanged=true
fi

# A commit made through the API is signed by GitHub for the token's identity.
if ! ${unchanged}; then
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
fi

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
