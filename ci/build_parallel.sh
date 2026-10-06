#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

setup_build_log() {
  # Buffer workflow group markers along with diagnostics so parallel builds cannot interleave them.
  exec > "$1" 2>&1
  printf 'Build started at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Buffered output receives its GitHub timestamp on replay, not when the build actually ran.
  trap 'build_status=$?; printf "Build finished at %s (exit %s)\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${build_status}"' EXIT
}

run_logged_build() (
  local log_file=$1
  shift
  setup_build_log "${log_file}"
  "$@"
)

wait_for_builds() {
  local status=0
  local pid log_file build_status
  # A bare wait hides child failures; wait for every child before publishing artifacts.
  while [[ $# -gt 0 ]]; do
    pid=$1
    log_file=$2
    shift 2
    build_status=0
    wait "${pid}" || build_status=$?
    printf '::group::%s (exit %s)\n' "${log_file}" "${build_status}"
    cat "${log_file}" || status=1
    printf '::endgroup::\n'
    if [[ "${build_status}" != 0 ]]; then
      status=1
    fi
  done
  return "${status}"
}
