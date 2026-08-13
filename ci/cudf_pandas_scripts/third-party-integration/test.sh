#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Common setup steps shared by Python test jobs

set -euo pipefail


extract_lib_from_dependencies_yaml() {
    local file=$1
    # Parse all keys in dependencies.yaml under the "files" section,
    # extract all the keys that start with "test_", and extract the rest
    extracted_libs="$(yq -o json "$file" | jq -rc '.files | with_entries(select(.key | contains("test_"))) | keys | map(sub("^test_"; ""))')"
    echo "$extracted_libs"
}

main() {
    local dependencies_yaml="$1"

    LIBS=$(extract_lib_from_dependencies_yaml "$dependencies_yaml")
    LIBS=${LIBS#[}
    LIBS=${LIBS%]}

    if [ "$RAPIDS_BUILD_TYPE" == "pull-request" ]; then
        rapids-logger "Downloading artifacts from this pr jobs"
        CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --cuda "$RAPIDS_CUDA_VERSION" --arch any)")
    fi

    ANY_FAILURES=0

    for lib in ${LIBS//,/ }; do
        lib=$(echo "$lib" | tr -d '""')
        echo "Running tests for library $lib"

        . /opt/conda/etc/profile.d/conda.sh
        # Check the value of RAPIDS_BUILD_TYPE
        if [ "$RAPIDS_BUILD_TYPE" == "pull-request" ]; then
            rapids-logger "Generate Python testing dependencies"
            rapids-dependency-file-generator \
                --config "$dependencies_yaml" \
                --output conda \
                --file-key "test_${lib}" \
                --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
                --prepend-channel "${CPP_CHANNEL}" \
                --prepend-channel "${PYTHON_CHANNEL}" \
                --prepend-channel "${PYTHON_NOARCH_CHANNEL}" | tee env.yaml
        else
            rapids-logger "Generate Python testing dependencies"
            rapids-dependency-file-generator \
                --config "$dependencies_yaml" \
                --output conda \
                --file-key "test_${lib}" \
                --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml
        fi

        rapids-mamba-retry env create --yes -f env.yaml -n test

        # Temporarily allow unbound variables for conda activation.
        set +u
        conda activate test
        set -u

        repo_root=$(git rev-parse --show-toplevel)
        TEST_DIR=${repo_root}/python/cudf/cudf_pandas_tests/third_party_integration_tests/tests

        rapids-print-env

        rapids-logger "Check GPU usage"
        nvidia-smi

        rapids-logger "pytest ${lib}"

        NUM_PROCESSES=8
        serial_libraries=(
            "tensorflow"
        )
        for serial_library in "${serial_libraries[@]}"; do
            if [ "${lib}" = "${serial_library}" ]; then
                NUM_PROCESSES=1
            fi
        done

        EXITCODE=0
        trap "EXITCODE=1" ERR
        set +e

        TEST_DIR=${TEST_DIR} \
        NUM_PROCESSES=${NUM_PROCESSES} \
            timeout 45m \
            ci/cudf_pandas_scripts/third-party-integration/run-library-tests.sh "${lib}"

        set -e
        rapids-logger "Test script exiting with value: ${EXITCODE}"
        if [[ ${EXITCODE} != 0 ]]; then
            ANY_FAILURES=1
        fi
    done

    exit ${ANY_FAILURES}
}

main "$@"
