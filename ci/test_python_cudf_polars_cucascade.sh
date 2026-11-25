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
# Run from the repository root so that neither pytest nor the Ray workers, which put the
# working directory on sys.path, import the extension's source directory instead of the
# installed extension.
pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-cucascade.xml" \
  --durations=20 \
  python/cudf_polars/cucascade/tests
