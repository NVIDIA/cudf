#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
CUDF_POLARS_BASELINE_SHA=$(git -C "${SCRIPT_DIR}/.." rev-parse HEAD)
export CUDF_POLARS_BASELINE_SHA
RUN_LIST_ARGS=()
if [[ -n "${CUDF_POLARS_GPU_RUN_LIST:-}" ]]; then
    RUN_LIST_ARGS=(--inject-gpu-engine-run-list="$(realpath "${SCRIPT_DIR}/../${CUDF_POLARS_GPU_RUN_LIST}")")
fi
if [[ "${CI:-false}" == "true" ]]; then
    # GitHub exposes the job name but not its numeric ID to steps; the audit
    # matches each report to that ID to detect missing matrix jobs and shards.
    producer_json=$(gh api --paginate \
        "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?filter=latest&per_page=100" \
        | jq -c --arg runner "${RUNNER_NAME}" \
            '.jobs[] | select(.runner_name == $runner and .status == "in_progress") | {id, configuration: (.name | split(" / ") | last)}')
    CUDF_POLARS_REPORT_PRODUCER=$(printf '%s' "${producer_json}" | jq -r .id)
    CUDF_POLARS_REPORT_CONFIGURATION=$(printf '%s' "${producer_json}" | jq -r .configuration)
    export CUDF_POLARS_REPORT_PRODUCER CUDF_POLARS_REPORT_CONFIGURATION
    if [[ ! "${CUDF_POLARS_REPORT_PRODUCER}" =~ ^[0-9]+$ ]]; then
        echo "Could not identify the unique GitHub test producer job" >&2
        exit 2
    fi
fi
TIMEOUT_TOOL_PATH="${SCRIPT_DIR}/timeout_with_stack.py"
test_timeout=5400
if [[ ${RAPIDS_BUILD_TYPE:-} == pull-request ]]; then
    test_timeout=1800
fi
DESELECTED_TESTS_FILE="${SCRIPT_DIR}/cudf_polars/polars_test_deselections.txt"
AARCH64_DESELECTED_TESTS_FILE="${SCRIPT_DIR}/cudf_polars/polars_test_deselections_aarch64.txt"
INCOMPATIBLE_GLIBC_DESELECTED_TESTS_FILE="${SCRIPT_DIR}/cudf_polars/polars_test_deselections_incompatible_glibc.txt"

function load_deselected_tests()
{
    local file="$1"
    local test
    while IFS= read -r test || [[ -n "${test}" ]]; do
        if [[ -n "${test}" && "${test}" != \#* ]]; then
            DESELECTED_TESTS+=("${test}")
        fi
    done < "${file}"
}

function configure_fallback_report()
{
    FALLBACK_REPORT_ARGS=()
    if [[ "${CI:-false}" == "true" && -n "${RAPIDS_ARTIFACTS_DIR:-}" ]]; then
        mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
        local report_id
        report_id=$(python -c 'import uuid; print(uuid.uuid4().hex)')
        # Matrix jobs can share engine, architecture, and Python/CUDA versions;
        # a unique suffix prevents the shared uploader overwriting their reports.
        FALLBACK_REPORT_ARGS=(
            --inject-gpu-engine-report="${RAPIDS_ARTIFACTS_DIR}/cudf-polars-fallback-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${1}-${report_id}.json"
        )
    fi
}

ENGINE="both"
BLOCKSIZE="default"
RUN_SLOW=false
PYTEST_ARGS=()
while (($#)); do
    case "$1" in
        --engine)
            if (($# < 2)); then
                echo "Missing value for --engine." >&2
                exit 2
            fi
            ENGINE="$2"
            shift 2
            ;;
        --engine=*)
            ENGINE="${1#*=}"
            shift
            ;;
        --inject-gpu-engine-blocksize)
            if (($# < 2)); then
                echo "Missing value for --inject-gpu-engine-blocksize." >&2
                exit 2
            fi
            BLOCKSIZE="$2"
            shift 2
            ;;
        --inject-gpu-engine-blocksize=*)
            BLOCKSIZE="${1#*=}"
            shift
            ;;
        --run-slow)
            RUN_SLOW=true
            shift
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${ENGINE}" != "both" && "${ENGINE}" != "in-memory" && "${ENGINE}" != "spmd" ]]; then
    echo "Unknown engine: ${ENGINE}. Expected one of: both, in-memory, spmd." >&2
    exit 2
fi
if [[ "${BLOCKSIZE}" != "default" && "${BLOCKSIZE}" != "small" ]]; then
    echo "Unknown blocksize: ${BLOCKSIZE}. Expected one of: default, small." >&2
    exit 2
fi
if [[ "${RUN_SLOW}" == "true" ]]; then
    PYTEST_MARK_EXPR=""
else
    PYTEST_MARK_EXPR="not slow"
fi

cd "${POLARS_DIR:-${SCRIPT_DIR}/../polars}"

DESELECTED_TESTS=()
load_deselected_tests "${DESELECTED_TESTS_FILE}"

if [[ $(arch) == "aarch64" ]]; then
    load_deselected_tests "${AARCH64_DESELECTED_TESTS_FILE}"
else
    # Ensure that we don't run dbgen when it uses newer symbols than supported by the glibc version in the CI image.
    # Allow errors since any of these commands could produce empty results that would cause the script to fail.
    set +e
    glibc_minor_version=$(ldd --version | head -1 | grep -o "[0-9]\.[0-9]\+" | tail -1 | cut -d '.' -f2)
    latest_glibc_symbol_found=$(nm py-polars/tests/benchmark/data/pdsh/dbgen/dbgen | grep GLIBC | grep -o "[0-9]\.[0-9]\+" | sort --version-sort | tail -1 | cut -d "." -f 2)
    set -e
    if [[ ${glibc_minor_version} -lt ${latest_glibc_symbol_found} ]]; then
        load_deselected_tests "${INCOMPATIBLE_GLIBC_DESELECTED_TESTS_FILE}"
    fi
fi

DESELECTED_TEST_ARGS=()
for test in "${DESELECTED_TESTS[@]}"; do
    DESELECTED_TEST_ARGS+=(--deselect "${test}")
done

# SQLAlchemy 2.1 warns about URI parameters used by the Polars 1.44 Iceberg
# tests. Polars fixed this in https://github.com/pola-rs/polars/pull/29533.
SQLALCHEMY_SQLITE_URI_WARNING="ignore:Query string argument(s) 'cache', 'mode' are not accepted by the pysqlite driver and are being ignored; SQLite URI arguments require that 'uri=true' also be present in the URL.:sqlalchemy.exc.SAWarning"

# Fail fast (-x) rather than trying to continue because failed tests pollute the state
if [[ "${ENGINE}" == "both" || "${ENGINE}" == "in-memory" ]]; then
    configure_fallback_report in-memory
    echo "Run polars tests with injected in-memory GPU engine"
    python "${TIMEOUT_TOOL_PATH}" --enable-python "${test_timeout}" \
       python -m pytest \
           --import-mode=importlib \
           --cache-clear \
           -x \
           -m "${PYTEST_MARK_EXPR}" \
           -p cudf_polars.testing.inject_gpu_engine \
           -n 4 \
           --dist=worksteal \
           --tb=native \
           --durations=50 --durations-min=1 \
           -W "${SQLALCHEMY_SQLITE_URI_WARNING}" \
           "${DESELECTED_TEST_ARGS[@]}" \
           "${PYTEST_ARGS[@]}" \
           "${FALLBACK_REPORT_ARGS[@]}" \
           "${RUN_LIST_ARGS[@]}" \
           py-polars/tests \
           --inject-gpu-engine in-memory
fi

# TODO(ResourceWarning): https://github.com/NVIDIA/cudf/issues/22181
if [[ "${ENGINE}" == "both" || "${ENGINE}" == "spmd" ]]; then
    configure_fallback_report spmd
    echo "Run polars tests with injected SPMD GPU engine, ${BLOCKSIZE} blocksize"
    CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE=805306368 \
    CUDF_POLARS__EXECUTOR__FALLBACK_MODE=silent \
    python "${TIMEOUT_TOOL_PATH}" --enable-python "${test_timeout}" \
       python -m pytest \
           --import-mode=importlib \
           --cache-clear \
           -x \
           -m "${PYTEST_MARK_EXPR}" \
           -p cudf_polars.testing.inject_gpu_engine \
           -W ignore::ResourceWarning \
           -n 4 \
           --dist=worksteal \
           --tb=native \
           --durations=50 --durations-min=1 \
           -W "${SQLALCHEMY_SQLITE_URI_WARNING}" \
           "${DESELECTED_TEST_ARGS[@]}" \
           "${PYTEST_ARGS[@]}" \
           "${FALLBACK_REPORT_ARGS[@]}" \
           "${RUN_LIST_ARGS[@]}" \
           py-polars/tests \
           --inject-gpu-engine spmd \
           --inject-gpu-engine-blocksize "${BLOCKSIZE}"
fi
