#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

PYDISTCHECK_ARGS=(
    --inspect
)

# PyPI hard limit is 1GiB, but try to keep these as small as possible
if [[ "${package_dir}" == "python/libcudf" ]]; then
    if [[ "${RAPIDS_CUDA_MAJOR}" == "12" ]]; then
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '700M'
        )
    else
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '350M'
        )
    fi
elif [[ "${package_dir}" == "python/libcudf_streaming" ]]; then
    PYDISTCHECK_ARGS+=(
        --max-allowed-size-compressed '100M'
    )
elif [[ "${package_dir}" == "python/cudf_streaming" ]]; then
    PYDISTCHECK_ARGS+=(
        --max-allowed-size-compressed '75M'
    )
elif [[ "${package_dir}" != "python/cudf" ]] && \
     [[ "${package_dir}" != "python/cudf_polars" ]] && \
     [[ "${package_dir}" != "python/dask_cudf" ]] && \
     [[ "${package_dir}" != "python/pylibcudf" ]]; then
    rapids-echo-stderr "unrecognized package_dir: '${package_dir}'"
    exit 1
fi

pydistcheck \
    "${PYDISTCHECK_ARGS[@]}" \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"
