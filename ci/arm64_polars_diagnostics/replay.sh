#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_REPOSITORY=NVIDIA/cudf
export RAPIDS_BASE_BRANCH=main
RAPIDS_SHA=$(git -C replay-source rev-parse HEAD)
export RAPIDS_SHA
export CUDF_POLARS_DIAGNOSTICS_DIR="$PWD/artifacts/arm64-polars"
export RAPIDS_TESTS_DIR="$CUDF_POLARS_DIAGNOSTICS_DIR/test-results"
export RAPIDS_COVERAGE_DIR="$CUDF_POLARS_DIAGNOSTICS_DIR/coverage"
{ read -r _; IFS=, read -r CUDF_POLARS_GPU_UUID _; } < "$CUDF_POLARS_DIAGNOSTICS_DIR/gpus.csv"
export CUDF_POLARS_GPU_UUID

export PYTHONPATH="$PWD/ci/arm64_polars_diagnostics${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_PLUGINS=repro_diagnostics
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-* ]]; then
    CUDF_POLARS_FAULT_LIBRARY=$(mktemp /tmp/cudf-polars-fault.XXXXXX.so)
    export CUDF_POLARS_FAULT_LIBRARY
    conda run --no-capture-output --name fault-compiler aarch64-conda-linux-gnu-gcc \
        -std=c11 -Wall -Wextra -O2 -fPIC -shared \
        ci/arm64_polars_diagnostics/fault_address.c -o "$CUDF_POLARS_FAULT_LIBRARY"
fi
if [[ "${CUDF_POLARS_DIAGNOSTICS_TRIAL}" == warmup-* ]]; then
    # Coverage gates do not apply to the selected worker history.
    export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --no-cov -W ignore::pytest_cov.CovDisabledWarning"
fi

# Pin both the test source and the artifact set so a rebuild cannot silently
# change the native binaries used for the aggregation comparison.
if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == *-aggregation ]]; then
    export RAPIDS_REF_NAME=pull-request/25384
    export CUDF_POLARS_PACKAGE_RUN_ID=45661654812
    export CUDF_POLARS_EXPLICIT_ENV="$PWD/ci/arm64_polars_diagnostics/aggregation-arm64-explicit.txt"
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == *-frozen-shuffle ]]; then
    export RAPIDS_REF_NAME=pull-request/25452
    export CUDF_POLARS_PACKAGE_RUN_ID=45449689051
    export CUDF_POLARS_EXPLICIT_ENV="$PWD/ci/arm64_polars_diagnostics/frozen-shuffle-arm64-explicit.txt"
else
    # The first shuffle failure's packages were rebuilt at the same source revision.
    export RAPIDS_REF_NAME=pull-request/25532
    export CUDF_POLARS_PACKAGE_RUN_ID=45518453978
    export CUDF_POLARS_EXPLICIT_ENV="$PWD/ci/arm64_polars_diagnostics/original-arm64-explicit.txt"
fi

rapids-download-from-github() {
    GITHUB_RUN_ID="$CUDF_POLARS_PACKAGE_RUN_ID" command rapids-download-from-github "$@"
}
export -f rapids-download-from-github

# Avoid changing nightly dependencies when the replay crosses daily builds.
rapids-mamba-retry() {
    local spec entry
    spec="$CUDF_POLARS_DIAGNOSTICS_DIR/requested-explicit.txt"
    while IFS= read -r entry; do
        entry=${entry/\{CPP_CHANNEL\}/file:\/\/${CPP_CHANNEL}}
        entry=${entry/\{PYTHON_CHANNEL\}/file:\/\/${PYTHON_CHANNEL}}
        entry=${entry/\{PYTHON_NOARCH_CHANNEL\}/file:\/\/${PYTHON_NOARCH_CHANNEL}}
        entry=${entry/\{CUDF_POLARS_QUENT_CHANNEL\}/file:\/\/${CUDF_POLARS_QUENT_CHANNEL}}
        printf '%s\n' "$entry"
        if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck-native-aggregation &&
            "$entry" == */libcudf-streaming-*.conda ]]; then
            printf '%s\n' "${entry/\/libcudf-streaming-/\/libcudf-streaming-tests-}"
        fi
    done < "$CUDF_POLARS_EXPLICIT_ENV" > "$spec"
    conda create --yes --no-default-packages --name test --file "$spec"
    # Conda's explicit manifest omits the wheel installed by the original job.
    conda run --no-capture-output --name test python -m pip install --no-deps 'polars-iceberg==0.1.1'
}
export -f rapids-mamba-retry

if [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-stress-aggregation ]]; then
    printf '%s\n' '#!/bin/bash' 'exec timeout --kill-after=15s 8m python -m stress_packed_size' \
        > replay-source/ci/run_cudf_polars_pytests.sh
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == signal-repeat-* ]]; then
    cp ci/arm64_polars_diagnostics/repeat-spmd-tests.sh replay-source/ci/run_cudf_polars_pytests.sh
elif [[ "$CUDF_POLARS_DIAGNOSTICS_TRIAL" == memcheck* ]]; then
    cp ci/arm64_polars_diagnostics/memcheck-pytest.sh replay-source/ci/run_cudf_polars_pytests.sh
elif [[ "$CUDF_POLARS_REPEAT_REPLAY" == true ]]; then
    # Resolve the exact packages once and run fresh pytest processes on the same GPU.
    mv replay-source/ci/run_cudf_polars_pytests{,-once}.sh
    cp ci/arm64_polars_diagnostics/repeat-pytest.sh replay-source/ci/run_cudf_polars_pytests.sh
    chmod +x replay-source/ci/run_cudf_polars_pytests.sh
fi
cd replay-source
env POLARS_VERSIONS=latest bash ci/test_python_cudf_polars.sh
