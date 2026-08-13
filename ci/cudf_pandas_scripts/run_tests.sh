#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# https://github.com/jupyter/jupyter_core/pull/292
export JUPYTER_PLATFORM_DIRS=1

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the cudf, libcudf, and pylibcudf built in the previous step
source rapids-prompt-local-repo-config
LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python_cudf_pandas ./constraints.txt

# notes:
#
#   * echo to expand wildcard before adding `[test,cudf-pandas-tests]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,cudf-pandas-tests]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

python -m ipykernel install --user --name python3

rapids-logger "pytest cudf.pandas parallel"
# The third-party integration tests are ignored because they are run in a separate nightly CI job
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --numprocesses=8 \
    --dist=worksteal \
    -k "not profiler" \
    -m "not serial" \
    --config-file=./python/cudf/pyproject.toml \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/


rapids-logger "pytest cudf.pandas serial"

python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    -k "not profiler" \
    -m "serial" \
    --config-file=./python/cudf/pyproject.toml \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage-serial.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/

# pytest-xdist and pytest-cov prevent our profiler's trace function from running,
# even with pytest.mark.no_cover. This likely stems from specialized logic in
# coveragepy and pytest-cov for distributed testing (pytest-dev/pytest-cov#246).
# As a workaround, we run profiler tests separately without parallelism or `--cov`.
# More details: https://github.com/rapidsai/cudf/pull/16930#issuecomment-2707873968
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --config-file=./python/cudf/pyproject.toml \
    --numprocesses=0 \
    -k "profiler" \
    ./python/cudf/cudf_pandas_tests/

read -r -a versions <<< "$(python ci/utils/get_matrix_values.py dependencies.yaml test_cudf_pandas_compat pandas_compat_version)"

for version in "${versions[@]}"; do
    rapids-logger "Testing cudf.pandas compatibility with pandas ${version}.*"

    # Generate requirements for this pandas compat version.
    # Each entry pins the specific pandas minor line and the CUDA-appropriate cupy.
    pandas_requirements_txt="pandas-compat-${version}-requirements.txt"
    rapids-dependency-file-generator \
        --config dependencies.yaml \
        --file-key test_cudf_pandas_compat \
        --output requirements \
        --matrix "cuda=${RAPIDS_CUDA_VERSION};pandas_compat_version=${version}" \
        > "${pandas_requirements_txt}"

    env_name="venv_pandas_${version}"
    python -m venv --clear "${env_name}"
    # shellcheck disable=SC1090
    source "${env_name}/bin/activate"

    # notes:
    #
    #   * echo to expand wildcard before adding `[test,cudf-pandas-tests]` requires for pip
    #   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
    #     ignored if any other --constraint are passed via the CLI
    #
    rapids-pip-retry install \
        -v \
        --constraint ./constraints.txt \
        --constraint "${PIP_CONSTRAINT}" \
        "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,cudf-pandas-tests]" \
        "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
        "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
        -r "${pandas_requirements_txt}"

    python -m pytest -p cudf.pandas \
        --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
        --numprocesses=8 \
        --dist=worksteal \
        -k "not profiler" \
        -m "not serial" \
        --config-file=./python/cudf/pyproject.toml \
        --cov-config=./python/cudf/.coveragerc \
        --cov=cudf \
        --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
        --cov-report=term \
        ./python/cudf/cudf_pandas_tests/

    # NOTE: We don't currently run serial tests (only 1 as of 2025-07-25)
    # with multiple versions of pandas.

    python -m pytest -p cudf.pandas \
        --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
        --numprocesses=0 \
        -k "profiler" \
        ./python/cudf/cudf_pandas_tests/

    deactivate
    rm -rf "${env_name}" "${pandas_requirements_txt}"
done
