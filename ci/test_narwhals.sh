#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_python_cudf.sh outside the script directory
CI_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
cd "${CI_DIR}/../"

source rapids-init-pip

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_narwhals

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest narwhals"
NARWHALS_VERSION=$(python -c "import narwhals; print(narwhals.__version__)")
git clone https://github.com/narwhals-dev/narwhals.git --depth=1 -b "v${NARWHALS_VERSION}" narwhals
pushd narwhals
rapids-pip-retry install -U -e . --no-build-isolation

rapids-logger "Check narwhals versions"
python -c "import narwhals; print(narwhals.show_versions())"

rapids-logger "Run narwhals tests for cuDF"
PYTHONPATH="${CI_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    timeout 15m \
    python -m pytest \
    --cache-clear \
    -p xdist \
    -p env \
    -p narwhals_cudf_test_plugin \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=cudf

rapids-logger "Run narwhals tests for cuDF Polars"
CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE=805306368 \
CUDF_POLARS__EXECUTOR__FALLBACK_MODE=silent \
PYTHONPATH="${CI_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NARWHALS_POLARS_GPU=1 \
    timeout 15m \
    python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-narwhals.xml" \
    -p xdist \
    -p env \
    -p narwhals_cudf_polars_test_plugin \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=polars[lazy]

rapids-logger "Run narwhals tests for cuDF Pandas"
PYTHONPATH="${CI_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NARWHALS_DEFAULT_CONSTRUCTORS=pandas \
    timeout 15m \
    python -m pytest \
    -p cudf.pandas \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas-narwhals.xml" \
    -p xdist \
    -p env \
    -p narwhals_cudf_pandas_test_plugin \
    --numprocesses=8 \
    --dist=worksteal

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
