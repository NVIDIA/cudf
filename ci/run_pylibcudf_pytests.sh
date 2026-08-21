#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Find the libcudf_identify_stream_usage_mode_testing.so library
TESTING_LIB=$(python <<'EOF'
import os

# Import pylibcudf to ensure libcudf.so is loaded
import pylibcudf

with open("/proc/self/maps") as f:
    for line in f:
        if "libcudf.so" in line:
            libcudf_lib_path = line.split()[-1]
            break
    else:
        raise RuntimeError(
            "Could not find libcudf.so in /proc/self/maps"
        )

# Search for the testing library in that directory
testing_lib_pattern = os.path.join(
    os.path.dirname(libcudf_lib_path),
    "libcudf_identify_stream_usage_mode_testing.so"
)
if os.path.isfile(testing_lib_pattern):
    print(testing_lib_pattern)
else:
    print("")
EOF
)

# It is essential to cd into python/pylibcudf/pylibcudf as `pytest-xdist` + `coverage` seem to work only at this directory level.
# Do this after determining TESTING_LIB to avoid import path issues

# Support invoking run_cudf_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/pylibcudf/

# If the stream testing library was found, split the tests into two passes.
echo "RAPIDS_CUDA_VERSION: $RAPIDS_CUDA_VERSION"
if [ -n "$TESTING_LIB" ] && [ -f "$TESTING_LIB" ] && [ "$RAPIDS_CUDA_VERSION" != "12.2.2" ]; then
    # run the stream tests without xdist to ease the burden on the compute-sanitizer checks
    LD_PRELOAD="$TESTING_LIB" PYLIBCUDF_STREAM_TESTING=1 pytest --cache-clear -m "not uses_custom_stream" --ignore="benchmarks" -p no:xdist tests
fi
pytest --cache-clear --ignore="benchmarks" "$@" tests
