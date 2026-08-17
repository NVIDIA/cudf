#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_POLARS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-polars cudf --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")

# Download libcudf_streaming and cudf_streaming built in the previous step
LIBCUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-streaming cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf_polars "${PIP_CONSTRAINT}"

read -r -a VERSIONS <<< "$(python ci/utils/get_matrix_values.py dependencies.yaml test_cudf_polars_compat polars_compat_version)"

if [[ "${POLARS_VERSIONS:-all}" == "endpoints" ]] && [[ ${#VERSIONS[@]} -ge 2 ]]; then
    VERSIONS=("${VERSIONS[0]}" "${VERSIONS[-1]}")
fi

LATEST_VERSION="${VERSIONS[-1]}"

# shellcheck disable=SC2317
function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

PASSED=()
FAILED=()

for version in "${VERSIONS[@]}"; do
    rapids-logger "Testing cudf_polars with polars ${version}.*"

    # Generate requirements for this polars compat version.
    polars_requirements_txt="polars-compat-${version}-requirements.txt"
    rapids-dependency-file-generator \
        --config dependencies.yaml \
        --file-key test_cudf_polars_compat \
        --output requirements \
        --matrix "polars_compat_version=${version}" \
        > "${polars_requirements_txt}"

    env_name="venv_polars_${version}"
    python -m venv --clear "${env_name}"
    # shellcheck disable=SC1090
    source "${env_name}/bin/activate"

    rapids-logger "Installing cudf_polars and its dependencies for polars ${version}.*"

    # notes:
    #
    #   * echo to expand wildcard before adding `[test]` requires for pip
    #   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
    #     that environment variable is ignored if any other --constraint are passed via the CLI
    #
    rapids-pip-retry install \
        -v \
        --prefer-binary \
        --constraint "${PIP_CONSTRAINT}" \
        "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,dask,ray]" \
        "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
        "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
        "$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
        "$(echo "${CUDF_STREAMING_WHEELHOUSE}"/cudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
        -r "polars-compat-${version}-requirements.txt"

    rapids-logger "Running tests for polars ${version}.*"

    if [ "${version}" == "${LATEST_VERSION}" ]; then
        COVERAGE_ARGS=(
            --cov=cudf_polars
            --cov-fail-under=100
            --cov-report=term-missing:skip-covered
            --cov-config=./pyproject.toml
        )
    else
        COVERAGE_ARGS=(--no-cov)
    fi

    # Fail fast (-x) rather than trying to continue because failed tests pollute the state
    ./ci/run_cudf_polars_pytests.sh \
        "${COVERAGE_ARGS[@]}" \
        --numprocesses=4 \
        --dist=worksteal \
        --durations 10 --durations-min 10 \
        -x \
        -ra \
        --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-${version}.xml"

    test_exitcode=$?
    deactivate
    rm -rf "${env_name}" "${polars_requirements_txt}"

    if [ ${test_exitcode} -ne 0 ]; then
        EXITCODE=1
        FAILED+=("${version}")
        rapids-logger "Tests failed for polars ${version}.*"
        break
    else
        PASSED+=("${version}")
        rapids-logger "Tests passed for polars ${version}.*"
    fi
done

trap ERR
set -e

rapids-logger "Polars test summary:"
rapids-logger "PASSED: ${PASSED[*]:-none}"
rapids-logger "FAILED: ${FAILED[*]:-none}"

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Testing FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Testing PASSED"
fi
exit ${EXITCODE}
