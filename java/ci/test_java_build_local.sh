#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Local end-to-end verification of the build workflow: builds the static
# libcudf install tree and the classifier JAR for both CUDA 12 and CUDA 13,
# then runs the decoupled gather step to assemble the combined
# Maven-repository layout. Mirrors what the java-build matrix + java-gather
# jobs in .github/workflows/build.yaml do in CI.
#
# Runs on either x86_64 or aarch64 hosts. Each invocation covers only the
# host architecture: on x86_64 it produces the "cuda12" and "cuda13"
# classifier JARs; on aarch64 it produces "cuda12-arm64" and "cuda13-arm64".
# The arm64 suffix is added automatically by the child build scripts (via
# pom.xml's classifier logic keyed off `uname -m`). To cover all four
# release classifiers, run this script once on each architecture.
#
# Both static libcudf builds run in parallel, and both JAR builds run in
# parallel (each uses a nested bind-mount over /repo/java/target inside the
# container to isolate Maven output).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/argparse.sh"

WORK_DIR=""
PARALLEL_LEVEL="$(nproc)"
CMAKE_CUDA_ARCHITECTURES=""

# CUDA versions built for both classifiers. Keep in sync with the java-build
# matrix in .github/workflows/build.yaml.
CUDA12_VERSION="12.9"
CUDA13_VERSION="13.3"

# Wall-clock timing variables.
STEP_NAMES=()
STEP_ELAPSED=()

print_help() {
  cat << EOF

Usage: test_java_build_local.sh --work-dir <path> [OPTIONS]

Runs the full build+gather pipeline for both CUDA 12 and CUDA 13 on the host
architecture (x86_64 or aarch64) and assembles the combined Maven-repository
layout.

REQUIRED:
    -w, --work-dir       Scratch directory for build outputs. Subtrees created:
                             <work-dir>/libcudf-cuda12    static libcudf (CUDA 12)
                             <work-dir>/libcudf-cuda13    static libcudf (CUDA 13)
                             <work-dir>/jars/<classifier> per-classifier JAR + POM
                             <work-dir>/maven-repo        combined Maven layout
                         where <classifier> is "cuda12" / "cuda13" on x86_64
                         and "cuda12-arm64" / "cuda13-arm64" on aarch64.

OPTIONS:
    -j, --parallel       Total build parallelism (default: nproc = ${PARALLEL_LEVEL}).
                         Each concurrent JAR build gets --parallel/2 to avoid
                         RAM pressure from two parallel nvcc runs.
    -A, --cmake-cuda-architectures
                         CUDA architecture list (e.g. "80" or "80;90") passed to
                         both build scripts. The literal value "all" is a
                         sentinel meaning "do not pass --cmake-cuda-architectures
                         to child scripts" — child scripts then fall back to
                         cuDF's default RAPIDS full architecture list (slow).
                         Default: auto-detect the local GPU's compute
                         capability via nvidia-smi (e.g. Ampere -> "80"). If
                         nvidia-smi is missing or returns nothing, falls back
                         to "all".
    -h, --help           Show this help message.

EXAMPLES:
    # Fast run (auto-detect local GPU arch):
    ./java/ci/test_java_build_local.sh --work-dir /tmp/java-build-test

    # Explicit override:
    ./java/ci/test_java_build_local.sh --work-dir /tmp/java-build-test \\
        --cmake-cuda-architectures 80

    # Full RAPIDS arch list (slow, e.g. GPU-less host):
    ./java/ci/test_java_build_local.sh --work-dir /tmp/java-build-test \\
        --cmake-cuda-architectures all

EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        print_help
        exit 0
        ;;
      -w|--work-dir)
        require_value "$1" "$2"
        WORK_DIR=$2
        shift 2
        ;;
      -j|--parallel)
        require_value "$1" "$2"
        PARALLEL_LEVEL=$2
        shift 2
        ;;
      -A|--cmake-cuda-architectures)
        require_value "$1" "$2"
        CMAKE_CUDA_ARCHITECTURES=$2
        shift 2
        ;;
      *)
        echo "Error: Unknown argument $1"
        print_help
        exit 1
        ;;
    esac
  done
}

log_step() {
  echo
  echo "============================================================"
  echo "== $1"
  echo "============================================================"
}

format_elapsed() {
  local s=$1
  printf '%dm %02ds' $((s/60)) $((s%60))
}

record_step_end() {
  local name=$1
  local start=$2
  local elapsed=$((SECONDS - start))
  STEP_NAMES+=("${name}")
  STEP_ELAPSED+=("${elapsed}")
  echo
  echo "== ${name} completed in $(format_elapsed "${elapsed}")"
}

parse_args "$@"

require_arg --work-dir "${WORK_DIR}"

mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Remove outputs from any prior run so cmake/mvn does not see stale artifacts.
rm -rf "${WORK_DIR}/libcudf-cuda12" \
       "${WORK_DIR}/libcudf-cuda13" \
       "${WORK_DIR}/jars" \
       "${WORK_DIR}/maven-repo"

# Auto-detect the local GPU's compute capability when the flag was not passed.
# "all" is the sentinel value that means "do not forward this flag to child
# scripts". Child scripts fall back to cuDF's default RAPIDS architecture
# list (slow but correct on GPU-less hosts).
if [[ -z ${CMAKE_CUDA_ARCHITECTURES} ]]; then
  DETECTED=""
  if command -v nvidia-smi > /dev/null 2>&1; then
    DETECTED=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | tr -d ' ')
  fi
  if [[ -n ${DETECTED} ]]; then
    CMAKE_CUDA_ARCHITECTURES=${DETECTED}
    echo "Auto-detected local GPU compute capability: ${CMAKE_CUDA_ARCHITECTURES}"
  else
    CMAKE_CUDA_ARCHITECTURES="all"
    echo "No local GPU detected; falling back to cuDF's default RAPIDS architecture list"
  fi
fi

echo "cuDF Java local build verification"
echo "  host arch:                  $(uname -m)"
echo "  work dir:                   ${WORK_DIR}"
echo "  parallel:                   ${PARALLEL_LEVEL}"
echo "  cuda12 version:             ${CUDA12_VERSION}"
echo "  cuda13 version:             ${CUDA13_VERSION}"
echo "  cmake cuda architectures:   ${CMAKE_CUDA_ARCHITECTURES}"
echo "  logs:                       ${LOG_DIR}/{static,jar}_cuda{12,13}.log"

# When forwarding to child scripts, "all" means "don't pass the flag".
CHILD_CMAKE_ARGS=()
if [[ ${CMAKE_CUDA_ARCHITECTURES} != "all" ]]; then
  CHILD_CMAKE_ARGS=(--cmake-cuda-architectures "${CMAKE_CUDA_ARCHITECTURES}")
fi

# Both Step 1 and Step 2 launch two concurrent builds. Each build gets half
# of PARALLEL_LEVEL so together they stay within PARALLEL_LEVEL.
STEP_PARALLEL=$((PARALLEL_LEVEL / 2))
if [[ ${STEP_PARALLEL} -lt 1 ]]; then
  STEP_PARALLEL=1
fi

# Step 1: static libcudf builds in parallel.
log_step "Step 1: building static libcudf for CUDA 12 and CUDA 13 in parallel"
STEP1_START=${SECONDS}

"${SCRIPT_DIR}/build_static_libcudf.sh" \
    --output-dir "${WORK_DIR}/libcudf-cuda12" \
    --cuda-version "${CUDA12_VERSION}" \
    --parallel "${STEP_PARALLEL}" \
    "${CHILD_CMAKE_ARGS[@]}" \
    > "${LOG_DIR}/static_cuda12.log" 2>&1 &
STATIC_CUDA12_PID=$!

"${SCRIPT_DIR}/build_static_libcudf.sh" \
    --output-dir "${WORK_DIR}/libcudf-cuda13" \
    --cuda-version "${CUDA13_VERSION}" \
    --parallel "${STEP_PARALLEL}" \
    "${CHILD_CMAKE_ARGS[@]}" \
    > "${LOG_DIR}/static_cuda13.log" 2>&1 &
STATIC_CUDA13_PID=$!

echo "  cuda12 static pid: ${STATIC_CUDA12_PID} (tail -f ${LOG_DIR}/static_cuda12.log)"
echo "  cuda13 static pid: ${STATIC_CUDA13_PID} (tail -f ${LOG_DIR}/static_cuda13.log)"

STATIC_CUDA12_RC=0
STATIC_CUDA13_RC=0
if ! wait "${STATIC_CUDA12_PID}"; then
  STATIC_CUDA12_RC=1
fi
if ! wait "${STATIC_CUDA13_PID}"; then
  STATIC_CUDA13_RC=1
fi

if [[ ${STATIC_CUDA12_RC} -ne 0 ]]; then
  echo "Error: static libcudf CUDA 12 build failed."
  echo "See ${LOG_DIR}/static_cuda12.log"
fi
if [[ ${STATIC_CUDA13_RC} -ne 0 ]]; then
  echo "Error: static libcudf CUDA 13 build failed."
  echo "See ${LOG_DIR}/static_cuda13.log"
fi
if [[ ${STATIC_CUDA12_RC} -ne 0 || ${STATIC_CUDA13_RC} -ne 0 ]]; then
  exit 1
fi

record_step_end "Step 1: static libcudf (parallel run)" "${STEP1_START}"

# Step 2: JAR builds in parallel. Each build's container nests a bind-mount
# over /repo/java/target so concurrent Maven runs don't clobber each other.
log_step "Step 2: packaging cuDF Java JARs for cuda12 and cuda13 in parallel"
STEP2_START=${SECONDS}

"${SCRIPT_DIR}/build_cudf_java_jar.sh" \
    --libcudf-dir "${WORK_DIR}/libcudf-cuda12" \
    --output-dir "${WORK_DIR}/jars" \
    --cuda-version "${CUDA12_VERSION}" \
    --parallel "${STEP_PARALLEL}" \
    "${CHILD_CMAKE_ARGS[@]}" \
    > "${LOG_DIR}/jar_cuda12.log" 2>&1 &
JAR_CUDA12_PID=$!

"${SCRIPT_DIR}/build_cudf_java_jar.sh" \
    --libcudf-dir "${WORK_DIR}/libcudf-cuda13" \
    --output-dir "${WORK_DIR}/jars" \
    --cuda-version "${CUDA13_VERSION}" \
    --parallel "${STEP_PARALLEL}" \
    "${CHILD_CMAKE_ARGS[@]}" \
    > "${LOG_DIR}/jar_cuda13.log" 2>&1 &
JAR_CUDA13_PID=$!

echo "  cuda12 jar pid: ${JAR_CUDA12_PID} (tail -f ${LOG_DIR}/jar_cuda12.log)"
echo "  cuda13 jar pid: ${JAR_CUDA13_PID} (tail -f ${LOG_DIR}/jar_cuda13.log)"

JAR_CUDA12_RC=0
JAR_CUDA13_RC=0
if ! wait "${JAR_CUDA12_PID}"; then
  JAR_CUDA12_RC=1
fi
if ! wait "${JAR_CUDA13_PID}"; then
  JAR_CUDA13_RC=1
fi

if [[ ${JAR_CUDA12_RC} -ne 0 ]]; then
  echo "Error: cuDF Java JAR CUDA 12 build failed."
  echo "See ${LOG_DIR}/jar_cuda12.log"
fi
if [[ ${JAR_CUDA13_RC} -ne 0 ]]; then
  echo "Error: cuDF Java JAR CUDA 13 build failed."
  echo "See ${LOG_DIR}/jar_cuda13.log"
fi
if [[ ${JAR_CUDA12_RC} -ne 0 || ${JAR_CUDA13_RC} -ne 0 ]]; then
  exit 1
fi

record_step_end "Step 2: JAR builds (parallel run)" "${STEP2_START}"

# Step 3: gather into a combined Maven-repository layout.
log_step "Step 3: assembling combined Maven-repository layout"
STEP3_START=${SECONDS}

"${SCRIPT_DIR}/assemble_maven_repo.sh" \
    --jars-dir "${WORK_DIR}/jars" \
    --output-dir "${WORK_DIR}/maven-repo"

record_step_end "Step 3: assemble Maven repo" "${STEP3_START}"

# Derive the assembled version from the output tree for display purposes.
CUDF_VERSION=$(basename "$(ls -d "${WORK_DIR}/maven-repo/ai/rapids/cudf"/*/ | head -1)")

log_step "Success"
echo "Combined Maven repository:"
echo "  ${WORK_DIR}/maven-repo/ai/rapids/cudf/${CUDF_VERSION}/"
echo
echo "Timings:"
for i in "${!STEP_NAMES[@]}"; do
  printf '  %-45s %s\n' "${STEP_NAMES[$i]}" "$(format_elapsed "${STEP_ELAPSED[$i]}")"
done
printf '  %-45s %s\n' "Total wall time" "$(format_elapsed "${SECONDS}")"
