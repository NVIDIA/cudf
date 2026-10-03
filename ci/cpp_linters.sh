#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key cpp_linters \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n cpp_linters

# Temporarily allow unbound variables for conda activation.
set +u
conda activate cpp_linters
set -u

# clang-tidy parses the GCC compile command with clang. Newer conda compilers add
# this GCC-only optimization flag, which clang reports as an error.
for flags_var in CFLAGS CXXFLAGS; do
  if [[ -n "${!flags_var:-}" ]]; then
    export "${flags_var}=$(printf '%s' "${!flags_var}" | sed -E 's/(^|[[:space:]])-fno-merge-constants([[:space:]]|$)/ /g; s/[[:space:]]+/ /g; s/^ //; s/ $//')"
  fi
done

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="cudf-cpp-linters-preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

source rapids-configure-sccache

sccache --stop-server 2>/dev/null || true

# Run the build via CMake, which will run clang-tidy when CUDF_STATIC_LINTERS is enabled.

iwyu_flag=""
if [[ "${RAPIDS_BUILD_TYPE:-}" == "nightly" || "${RAPIDS_BUILD_TYPE:-}" == "pull-request" ]]; then
  diagnostics_dir="${RAPIDS_ARTIFACTS_DIR:-${PWD}/artifacts}/iwyu-diagnostics"
  mkdir -p "${diagnostics_dir}"

  export CUDF_IWYU_LOG_FILE="${diagnostics_dir}/invocations.log"
  export CUDF_IWYU_DIAGNOSTICS_DIR="${diagnostics_dir}"
  iwyu_real_exe="$(command -v include-what-you-use)"
  export CUDF_IWYU_REAL_EXE="${iwyu_real_exe}"
  # Trace only the invocations that remained active in the first diagnostic run.
  if strace -o "${diagnostics_dir}/strace-probe.log" /bin/true; then
    export CUDF_IWYU_STRACE_SOURCE_REGEX='(expression_parser|binaryop)\.cpp$'
  else
    printf 'strace is unavailable under this runner security policy\n' >> "${diagnostics_dir}/strace-probe.log"
  fi
  # Profiling at launch captures the CPU-bound interval that strace cannot explain.
  if perf record --output "${diagnostics_dir}/perf-probe.data" -- /bin/true \
    >"${diagnostics_dir}/perf-probe.log" 2>&1; then
    export CUDF_IWYU_PERF_AVAILABLE=1
  else
    printf 'perf record is unavailable under this runner security policy\n' >> "${diagnostics_dir}/perf-probe.log"
  fi
  if gdb --batch --quiet --ex quit \
    >"${diagnostics_dir}/gdb-probe.log" 2>&1; then
    export CUDF_IWYU_GDB_SOURCE_REGEX='(expression_parser|binaryop)\.cpp$'
    export CUDF_IWYU_GDB_TIMEOUT_SECONDS=60
  else
    printf 'gdb is unavailable in the linter environment\n' >> "${diagnostics_dir}/gdb-probe.log"
  fi
  # A bounded failure lets the workflow upload diagnostics that cancellation skips.
  export CUDF_IWYU_TRACE_TIMEOUT_SECONDS=300
  iwyu_flag="-DCUDF_IWYU=ON -DIWYU_EXE=${PWD}/ci/iwyu_wrapper.sh"

  snapshot_iwyu_processes() {
    local pid

    printf '\n===== %s =====\n' "$(date -u +%FT%TZ)"
    for pid in $(pgrep -f include-what-you-use || true); do
      printf '\n----- PID %s -----\n' "${pid}"
      ps -p "${pid}" -o pid,ppid,stat,etime,wchan:32,args || true
      for proc_file in status stack syscall; do
        if [[ -r "/proc/${pid}/${proc_file}" ]]; then
          printf '\n/proc/%s/%s\n' "${pid}" "${proc_file}"
          cat "/proc/${pid}/${proc_file}" || true
        fi
      done
    done
  }

  (
    while sleep 60; do
      snapshot_iwyu_processes
    done
  ) | tee -a "${diagnostics_dir}/watchdog.log" &
  watchdog_pid=$!
  trap 'kill "${watchdog_pid}" 2>/dev/null || true; wait "${watchdog_pid}" 2>/dev/null || true' EXIT
fi
rapids-telemetry-record cpp_linters_build.log cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCUDF_CLANG_TIDY=ON ${iwyu_flag} -DBUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=75 -GNinja
if [[ -n "${iwyu_flag}" ]]; then
  cmake --build cpp/build --verbose 2>&1 | tee "${diagnostics_dir}/iwyu-build.log" | python cpp/scripts/parse_iwyu_output.py
else
  cmake --build cpp/build 2>&1 | python cpp/scripts/parse_iwyu_output.py
fi

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# Remove invalid components of the path for local usage. The path below is
# valid in the CI due to where the project is cloned, but presumably the fixes
# will be applied locally from inside a clone of cudf.
sed -i 's/\/__w\/cudf\/cudf\///' iwyu_results.txt
