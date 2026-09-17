#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
TIMEOUT_TOOL_PATH="${SCRIPT_DIR}/timeout_with_stack.py"
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

# Support invoking run_cudf_polars_pytests.sh outside the script directory
# Assumption, polars has been cloned in the root of the repo.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../polars/

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

# Fail fast (-x) rather than trying to continue because failed tests pollute the state
if [[ "${ENGINE}" == "both" || "${ENGINE}" == "in-memory" ]]; then
    echo "Run polars tests with injected in-memory GPU engine"
    python "${TIMEOUT_TOOL_PATH}" --enable-python 5400 \
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
           "${DESELECTED_TEST_ARGS[@]}" \
           "${PYTEST_ARGS[@]}" \
           py-polars/tests \
           --inject-gpu-engine in-memory
fi

# TODO(ResourceWarning): https://github.com/NVIDIA/cudf/issues/22181
if [[ "${ENGINE}" == "both" || "${ENGINE}" == "spmd" ]]; then
    echo "Run polars tests with injected SPMD GPU engine, ${BLOCKSIZE} blocksize"
    CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE=805306368 \
    CUDF_POLARS__EXECUTOR__FALLBACK_MODE=silent \
    python "${TIMEOUT_TOOL_PATH}" --enable-python 5400 \
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
           "${DESELECTED_TEST_ARGS[@]}" \
           "${PYTEST_ARGS[@]}" \
           py-polars/tests \
           --inject-gpu-engine spmd \
           --inject-gpu-engine-blocksize "${BLOCKSIZE}"
fi
