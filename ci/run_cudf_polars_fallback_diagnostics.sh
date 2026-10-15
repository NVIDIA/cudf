#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

report_dir=$(mktemp -d)
output_dir="${1:-cudf-polars-fallback-diagnostics}"

# The shared test workflows upload RAPIDS_ARTIFACTS_DIR even after test failures.
# Restrict downloads to this attempt so reruns cannot reuse earlier diagnostics.
aws s3 cp "$(rapids-s3-path)" "${report_dir}/" --recursive \
    --exclude '*' \
    --include "*.cudf-polars-fallback-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-*.json" \
    --only-show-errors

python ci/analyze_cudf_polars_fallbacks.py \
    --reports-dir "${report_dir}" \
    --output-dir "${output_dir}" \
    --summary-file "${GITHUB_STEP_SUMMARY}"
