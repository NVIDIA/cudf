#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_python_common.sh test_cudf_polars_cucascade

rapids-logger "Check GPU usage"
nvidia-smi

# The extension subclasses libcudf and pylibcudf types, so it is not packaged: it is built
# here against the libcudf and pylibcudf under test, as python/cudf_polars/cucascade/README.md
# describes.
rapids-logger "Build cudf-polars-cucascade"
python -m pip install --no-build-isolation --no-deps -v ./python/cudf_polars/cucascade

rapids-logger "pytest cudf-polars-cucascade"
# The pytest entry point, unlike `python -m pytest`, does not put the working directory on
# sys.path, so the tests import the installed extension rather than its source directory.
cd python/cudf_polars/cucascade
pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-cucascade.xml" \
  --durations=20
