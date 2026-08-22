#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_POLARS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-polars cudf --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "$RAPIDS_CUDA_VERSION")")
CUDF_STREAMING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-streaming cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-generate-pip-constraints py_test_cudf_polars "${PIP_CONSTRAINT}"

rapids-logger "Installing cudf_polars and TPC test dependencies"

TPCH_REQUIREMENTS=$(mktemp --suffix=.txt)
rapids-dependency-file-generator \
    --config dependencies.yaml \
    --file-key test_cudf_polars_tpch \
    --output requirements \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
    > "${TPCH_REQUIREMENTS}"

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${CUDF_STREAMING_WHEELHOUSE}"/cudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    -r "${TPCH_REQUIREMENTS}"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Generating TPC-H data at SF=1"

export TPCH_DATA_DIR
TPCH_DATA_DIR=$(mktemp -d)
python3 "$(dirname "$0")/generate_tpch_data.py" --scale 1 --output-dir "${TPCH_DATA_DIR}"

rapids-logger "Generating TPC-DS data at SF=1"

export TPCDS_DATA_DIR
TPCDS_DATA_DIR=$(mktemp -d)
python3 "$(dirname "$0")/generate_tpcds_data.py" --scale 1 --output-dir "${TPCDS_DATA_DIR}"

rapids-logger "Running TPC-H and TPC-DS validation tests"

cd python/cudf_polars

python -m pytest tests/streaming/test_tpch.py tests/streaming/test_tpcds.py \
    --iterations 2 \
    -v
