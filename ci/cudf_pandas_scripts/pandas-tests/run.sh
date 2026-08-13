#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Check GPU usage"
nvidia-smi

PANDAS_TESTS_BRANCH=${1}
RAPIDS_FULL_VERSION=$(<./VERSION)
rapids-logger "Running Pandas tests using $PANDAS_TESTS_BRANCH branch and rapids-version $RAPIDS_FULL_VERSION"
rapids-logger "PR number: ${RAPIDS_REF_NAME:-"unknown"}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the cudf, libcudf, and pylibcudf built in the previous step
LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
  "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,pandas-tests]" \
  "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

timeout 90m bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
  --durations=10 \
  --numprocesses 8 \
  --tb=line \
  --max-worker-restart=3 \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas.xml" \
  --dist worksteal \
  --report-log="${PANDAS_TESTS_BRANCH}.json" 2>&1

SUMMARY_FILE_NAME=${PANDAS_TESTS_BRANCH}-results.json
# summarize the results and save them to artifacts:
python python/cudf/cudf/pandas/scripts/summarize-test-results.py --output json pandas-testing/"${PANDAS_TESTS_BRANCH}.json" > "./${SUMMARY_FILE_NAME}"

# Exit early if running tests for main branch
if [[ "${PANDAS_TESTS_BRANCH}" == "main" ]]; then
    rapids-logger "Exiting early for main branch testing: ${EXITCODE}"
    exit ${EXITCODE}
fi


MAIN_RUN_ID=$(
    gh run list                       \
        -w "Pandas Test Job"          \
        -b "$(<./RAPIDS_BRANCH)"      \
        --repo 'rapidsai/cudf'        \
        --status success              \
        --limit 7                     \
        --json 'createdAt,databaseId' \
        --jq 'sort_by(.createdAt) | reverse | .[0] | .databaseId'
)

if [[ -z "${MAIN_RUN_ID}" ]]; then
    rapids-logger "No MAIN_RUN_ID found, exiting."
    exit ${EXITCODE}
fi

rapids-logger "Fetching latest available results from nightly: ${MAIN_RUN_ID}"
gh run download                  \
    --repo 'rapidsai/cudf'        \
    --name main-results.json \
    $MAIN_RUN_ID

# Compute the diff and prepare job summary:
python ci/cudf_pandas_scripts/pandas-tests/job-summary.py main-results.json pr-results.json "${RAPIDS_FULL_VERSION}" >> "$GITHUB_STEP_SUMMARY"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
