#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
RUN_CUDF="${RUN_CUDF:-true}"
RUN_CUDF_STREAMING="${RUN_CUDF_STREAMING:-true}"

# Download the wheel artifacts built in the previous steps.
LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

if [[ "${RUN_CUDF}" == "true" ]]; then
  CUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

  rapids-logger "Install libcudf and verify its runtime dependencies in a virtual environment"

  # Generate constraints (possibly pinning to oldest support versions of dependencies).
  rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

  RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
  RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
  mkdir -p "${RAPIDS_TESTS_DIR}"

  python -m venv libcudf-env
  . libcudf-env/bin/activate

  rapids-pip-retry install \
      -v \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
  python -c "import libcudf; assert libcudf.load_library() is not None"
  deactivate

  # To test pylibcudf without its optional dependencies, we create a virtual environment.
  python -m venv pylibcudf-env
  . pylibcudf-env/bin/activate

  rapids-logger "Install pylibcudf and its basic dependencies"

  # Notes:
  #
  #   * echo to expand wildcard before adding `[test]` requires for pip
  #   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
  #     that environment variable is ignored if any other --constraint are passed via the CLI
  #
  rapids-pip-retry install \
      -v \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
      "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

  rapids-logger "pytest pylibcudf without optional dependencies"
  pushd python/pylibcudf/tests
  timeout 30m python -m pytest \
    --cache-clear \
    --numprocesses=8 \
    --dist=worksteal \
    .
  popd

  deactivate

  rapids-logger "Install cudf, pylibcudf, and test requirements"

  rapids-pip-retry install \
      -v \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
      "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
      "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test, pyarrow, numpy]"

  rapids-logger "pytest pylibcudf"
  pushd python/pylibcudf/tests
  timeout 30m python -m pytest \
    --cache-clear \
    --numprocesses=8 \
    --dist=worksteal \
    .
  popd

  rapids-logger "pytest cudf"
  pushd python/cudf/cudf/tests
  timeout 30m python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
    --numprocesses=8 \
    --dist=worksteal \
    .
  popd
fi

if [[ "${RUN_CUDF_STREAMING}" == "true" ]]; then
  CUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-streaming cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
  LIBCUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "$RAPIDS_CUDA_VERSION")")

  # Generate constraints (possibly pinning to oldest support versions of dependencies).
  rapids-generate-pip-constraints py_test_cudf_streaming "${PIP_CONSTRAINT}" constraints

  rapids-logger "Install libcudf_streaming and verify its runtime dependencies in a virtual environment"

  python -m venv libcudf-streaming-env
  . libcudf-streaming-env/bin/activate

  rapids-pip-retry install \
      -v \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
      "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
  python -c "import libcudf_streaming; assert libcudf_streaming.load_library() is not None"
  deactivate

  rapids-logger "Install cudf_streaming and its dependencies"

  rapids-pip-retry install \
      -v \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(echo "${CUDF_STREAMING_WHEELHOUSE}"/cudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
      "$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
      "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
      "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

  rapids-logger "pytest cudf_streaming"
  pushd python/cudf_streaming/cudf_streaming/tests
  EXITCODE=0
  timeout 30m python -m pytest \
    --cache-clear \
    --numprocesses=8 \
    --dist=worksteal \
    . || EXITCODE=$?

  # Exit code 5 means no tests were collected (all skipped); acceptable when
  # communicator support (MPI/UCXX) is unavailable in the wheel test environment.
  if [ ${EXITCODE} -ne 0 ] && [ ${EXITCODE} -ne 5 ]; then
    exit ${EXITCODE}
  fi
  popd
fi
