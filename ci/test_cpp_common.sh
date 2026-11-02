#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# CI provides LIBCUDF_KERNEL_CACHE_PATH through the reusable workflow's cache-environment input.
# Resolve a workspace-relative value before CTest changes its working directory.
LIBCUDF_KERNEL_CACHE_PATH="${LIBCUDF_KERNEL_CACHE_PATH:-.cache/libcudf}"
if [[ "${LIBCUDF_KERNEL_CACHE_PATH}" != /* ]]; then
  LIBCUDF_KERNEL_CACHE_PATH="$(realpath -m "${LIBCUDF_KERNEL_CACHE_PATH}")"
fi
export LIBCUDF_KERNEL_CACHE_PATH
mkdir -p "${LIBCUDF_KERNEL_CACHE_PATH}"

rapids-logger "Scheduling experiment runner budget"
python - <<'PY'
import json
import os
import platform
from pathlib import Path

paths = [
    "/proc/self/cgroup",
    "/sys/fs/cgroup/cpu.max",
    "/sys/fs/cgroup/cpu.stat",
    "/sys/fs/cgroup/cpuset.cpus.effective",
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    "/sys/fs/cgroup/cpuset/cpuset.cpus",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
]
budget = {
    "architecture": platform.machine(),
    "cpu_info": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
    "visible_cpu_count": os.cpu_count(),
    "affinity": sorted(os.sched_getaffinity(0)),
    "memory_info": [
        line for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:"))
    ],
    "host_workers": os.environ.get("LIBCUDF_NUM_HOST_WORKERS", "default"),
    "allocator": os.environ.get("GTEST_CUDF_RMM_MODE", "pool"),
    "cgroup": {},
}
for name in paths:
    path = Path(name)
    if path.is_file():
        budget["cgroup"][name] = path.read_text().strip()
print("CUDF_SCHEDULING_RUNNER_BUDGET=" + json.dumps(budget, sort_keys=True))
PY
conda list --explicit

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi
