#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Common setup steps shared by Python test jobs

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Python testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

CMD="rapids-dependency-file-generator --output conda"

# Add file-key options for each argument
for KEY in "$@"; do
  CMD="${CMD} --file-key \"${KEY}\""
done

DEPENDENCY_MATRIX="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}"
if [[ -n "${CUDF_EXTRA_DEPENDENCY_MATRIX:-}" ]]; then
  DEPENDENCY_MATRIX="${DEPENDENCY_MATRIX};${CUDF_EXTRA_DEPENDENCY_MATRIX}"
fi

CMD="${CMD} \
  --prepend-channel \"${CPP_CHANNEL}\" \
  --prepend-channel \"${PYTHON_CHANNEL}\" \
  --prepend-channel \"${PYTHON_NOARCH_CHANNEL}\" \
  --matrix \"${DEPENDENCY_MATRIX}\""

eval ${CMD} | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${RESULTS_DIR}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"


function preserve_junit_reports()
{
    # Copy generated JUnit reports into $RAPIDS_ARTIFACTS_DIR so the "Upload
    # additional artifacts" step uploads it. This needs to run even when the
    # test suite fails.
    if [[ "${CI:-false}" != "true" || -z "${RAPIDS_ARTIFACTS_DIR:-}" ]]; then
        return
    fi
    local xml_file
    for xml_file in "${RAPIDS_TESTS_DIR}"/*.xml; do
        [[ -f "${xml_file}" ]] || continue
        mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
        local report_id
        report_id=$(python -c 'import uuid; print(uuid.uuid4().hex)')
        cp "${xml_file}" \
            "${RAPIDS_ARTIFACTS_DIR}/$(basename "${xml_file}" .xml)-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${report_id}.xml"
    done
}

rapids-print-env
