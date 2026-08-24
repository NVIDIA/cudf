#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-streaming cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_POLARS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-polars cudf --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf_polars "${PIP_CONSTRAINT}"

rapids-logger "Install libcudf, pylibcudf and cudf_polars"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${CUDF_STREAMING_WHEELHOUSE}"/cudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

TAG=$(python -c 'import polars; print(f"py-{polars.__version__}")')
rapids-logger "Clone polars to ${TAG}"
git clone https://github.com/pola-rs/polars.git --branch "${TAG}" --depth 1

# Install requirements for running polars tests
rapids-logger "Install polars test requirements"
# We don't need to pick up dependencies from polars-cloud, so we remove it.
sed -i '/^polars-cloud$/d' polars/py-polars/requirements-dev.txt
# Pin pyarrow to avoid FutureWarning https://github.com/pola-rs/polars/pull/28323.
sed -i 's/^pyarrow$/pyarrow<25/' polars/py-polars/requirements-dev.txt

# notes:
#
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    -r polars/py-polars/requirements-dev.txt \
    -r polars/py-polars/requirements-ci.txt

# shellcheck disable=SC2317
function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

rapids-logger "Run polars tests"
./ci/run_cudf_polars_polars_tests.sh

trap ERR
set -e

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Running polars test suite FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Running polars test suite PASSED"
fi
exit ${EXITCODE}
