#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

report_dir=$(mktemp -d)
audit_args=()
publish_audit=false
fetch_status=0
if [[ "${1:-}" == "--audit" || "${1:-}" == "--publish-audit" ]]; then
    [[ "$1" == "--publish-audit" ]] && publish_audit=true
    shift
    if ! gh api --paginate --slurp \
        "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?filter=latest&per_page=100" \
        | jq '[.[].jobs[] | select(.name | startswith("cudf-polars-polars-tests"))]' \
        > "${report_dir}/expected-jobs"; then
        fetch_status=1
    fi
    audit_args=(--audit-run-list ci/cudf_polars/gpu_test_run_list.json --expected-jobs "${report_dir}/expected-jobs")
fi
output_dir="${1:-cudf-polars-fallback-diagnostics}"

# The shared test workflows upload RAPIDS_ARTIFACTS_DIR even after test failures.
# Restrict downloads to this attempt so reruns cannot reuse earlier diagnostics.
if ! aws s3 cp "$(rapids-s3-path)" "${report_dir}/" --recursive \
    --exclude '*' \
    --include "*.cudf-polars-fallback-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-*.json" \
    --only-show-errors; then
    fetch_status=1
fi
if ((fetch_status != 0 && ${#audit_args[@]} != 0)); then
    # Partial downloads cannot establish membership or justify removals.
    printf '[]\n' > "${report_dir}/expected-jobs"
fi

audit_status=0
python ci/analyze_cudf_polars_fallbacks.py \
    --reports-dir "${report_dir}" \
    --output-dir "${output_dir}" \
    --summary-file "${GITHUB_STEP_SUMMARY}" \
    "${audit_args[@]}" || audit_status=$?
if ((fetch_status != 0)); then
    audit_status=1
fi

# The shared workflow uploads artifacts only after a successful script step.
# Nightlies publish first, then gate on the uploaded audit in a dependent job.
if ! ${publish_audit}; then
    exit "${audit_status}"
fi
