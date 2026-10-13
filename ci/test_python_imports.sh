#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading built Conda packages"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Install built packages with only their runtime dependencies"
rapids-mamba-retry create --yes --name python_imports --strict-channel-priority \
    --override-channels \
    --channel "$PYTHON_CHANNEL" --channel "$CPP_CHANNEL" \
    --channel rapidsai-nightly --channel conda-forge \
    "python=${RAPIDS_PY_VERSION}" "${PYTHON_CHANNEL}"/linux-64/*.conda

set +u
conda activate python_imports
set -u
rapids-print-env

for package in pylibcudf cudf cudf_kafka cudf_streaming; do
    rapids-logger "Import ${package} on a CPU-only runner"
    python -c "import ${package}"
done

rapids-logger "Check cudf.pandas CLI with initialization disabled"
RAPIDS_NO_INITIALIZE=1 python -m cudf.pandas -c "import pandas as pd; print(pd.Series([1, 2, 3]))"
CUDF_NO_INITIALIZE=1 python -m cudf.pandas -c "import pandas as pd; print(pd.Series([1, 2, 3]))"
