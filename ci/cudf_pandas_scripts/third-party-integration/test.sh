#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

INTEGRATION_RUN_DIR=''

cleanup() {
    local status=$?
    if [[ -n $INTEGRATION_RUN_DIR ]]; then
        rm -rf -- "$INTEGRATION_RUN_DIR" || status=1
    fi
    return "$status"
}

prepare_environment() {
    local dependencies_yaml=$1 environment=$2
    shift 2
    local lib channels=() file_keys=()
    for lib in "$@"; do
        file_keys+=(--file-key "test_${lib}")
    done
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        channels=(--prepend-channel "$CPP_CHANNEL" --prepend-channel "$PYTHON_CHANNEL" --prepend-channel "$PYTHON_NOARCH_CHANNEL")
    fi
    rapids-dependency-file-generator \
        --config "$dependencies_yaml" --output conda "${file_keys[@]}" \
        --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
        "${channels[@]}" | tee "${environment}.yaml" || return
    rapids-mamba-retry env create --yes \
        -f "${environment}.yaml" --prefix "$environment"
}

main() {
    local dependencies_yaml=$1 scripts_dir
    scripts_dir=$(dirname "${BASH_SOURCE[0]}")
    local library_json
    library_json=$(yq -o json "$dependencies_yaml" | jq -c '.files | keys | map(select(startswith("test_")) | sub("^test_"; ""))')
    local libraries=()
    mapfile -t libraries < <(jq -r '.[]' <<< "$library_json")
    if ((${#libraries[@]} == 0)); then
        printf 'No integration libraries selected.\n' >&2
        return 1
    fi
    # Partition this list when libraries require incompatible dependencies or isolation.
    local test_groups=("${libraries[*]}")
    local test_timeout=45m
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        test_timeout=15m
        CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --cuda "$RAPIDS_CUDA_VERSION" --arch any)")
    fi
    INTEGRATION_RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cudf-pandas-integration.XXXXXX")
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    rapids-logger "Check GPU availability"
    if ! nvidia-smi; then
        printf 'GPU availability check failed.\n' >&2
        return 1
    fi

    local status=0 test_status group group_index=0 environment
    local group_libraries=()
    for group in "${test_groups[@]}"; do
        read -r -a group_libraries <<< "$group"
        environment="${INTEGRATION_RUN_DIR}/group-${group_index}"
        group_index=$((group_index + 1))
        printf 'Prepare integration environment: %s\n' "$group"
        if prepare_environment "$dependencies_yaml" "$environment" "${group_libraries[@]}"; then
            printf 'Integration tests: %s\n' "$group"
            if timeout "${test_timeout}" bash "${scripts_dir}/run-library-tests.sh" "$environment" "${group_libraries[@]}"; then
                test_status=0
            else
                test_status=$?
                status=1
            fi
            printf 'INTEGRATION_GROUP libraries="%s" status=%s\n' "$group" "$test_status"
            printf 'Remove integration environment: %s\n' "$group"
            if ! rapids-mamba-retry env remove --yes \
                --prefix "$environment"; then
                printf 'Failed to remove group environment: %s\n' "$group" >&2
                return 1
            fi
        else
            printf 'Failed to prepare group environment: %s\n' "$group" >&2
            status=1
        fi
    done
    return "$status"
}

main "$@"
