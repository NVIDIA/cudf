#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2
# Match pydistcheck's default when callers do not provide a package-specific limit.
max_allowed_size_compressed=${3:-50M}

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

PYDISTCHECK_ARGS=(
    --inspect
    --max-allowed-size-compressed "${max_allowed_size_compressed}"
)

pydistcheck \
    "${PYDISTCHECK_ARGS[@]}" \
    "${wheel_dir_relative_path}"/*.whl

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "${wheel_dir_relative_path}"/*.whl

rapids-logger "validate packages with 'abi3audit'"

# 'abi3audit' fails on wheels with DSOs that lack an ABI tag (e.g. 'lib*' wheels).
# Filtering by '*abi*' avoids those.
find \
    "${wheel_dir_relative_path}" \
    -type f \
    -name '*abi*' \
    -exec abi3audit --strict --summary --verbose '{}' \+
